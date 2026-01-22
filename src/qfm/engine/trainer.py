import os

import torch
import torch.nn as nn
import torch.optim as optim
from ema_pytorch import EMA
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from qfm.config import cfg
from qfm.core.logger import logger
from qfm.dataset import BucketedBatchSampler, LatentDataset
from qfm.engine.inference import run_inference
from qfm.model_moe import MiniFluxDiT


def run_training():
    # === 1. 初始化 WandB ===
    wandb.init(
        project="QFM-Small-MoE",
        config={
            "lr": cfg.train.lr,
            "batch_size": cfg.train.batch_size,
            "image_count": 100000,
            "model": "MiniFluxDiT-MoE",
        },
    )

    # 硬件设置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = cfg.get_device()
    dtype = cfg.get_dtype(device)

    logger.info(f"🚀 Starting Training on {device} ({dtype})")
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    # === 2. 初始化模型 ===
    model = MiniFluxDiT(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        num_experts=cfg.model.num_experts,  # 8个专家
        qwen_dim=cfg.model.qwen_dim,
        clip_dim=cfg.model.clip_dim,
    ).to(device, dtype=dtype)

    # === 🔥 3. 初始化 EMA ===
    # EMA 会维护一个“影子模型”，它的参数是历史参数的平滑平均
    # beta=0.9999 意味着它非常平滑，抗干扰能力强
    ema = EMA(
        model,
        beta=0.9999,
        update_after_step=100,  # 前100步不稳定，不更新EMA
        update_every=10,  # 每10步更新一次，节省算力
    ).to(device)

    # 数据加载
    dataset = LatentDataset(cfg.PROCESSED_DATA_DIR)
    sampler = BucketedBatchSampler(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # === 🔥 4. 初始化 Warmup ===
    # 在前 2000 步，学习率从 1% 慢慢增加到 100%
    # 这对 MoE 至关重要，防止 Router 刚开始就瞎猜导致崩坍
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=2000)

    global_step = 0
    model.train()

    for epoch in range(cfg.train.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")

        for _step, batch in enumerate(pbar):
            x_1 = batch["x"].to(device, dtype=dtype)
            clip_vec = batch["t_embed"].to(device, dtype=dtype)
            qwen_emb = batch["context"].to(device, dtype=dtype)

            # --- 训练逻辑 ---
            batch_size = x_1.shape[0]
            t = torch.rand(batch_size, device=device, dtype=dtype)
            x_0 = torch.randn_like(x_1)
            t_expand = t.view(batch_size, 1, 1, 1)
            x_t = t_expand * x_1 + (1 - t_expand) * x_0
            v_target = x_1 - x_0

            # 强制类型转换 (修复推理时的 bug 这里也顺便加上保险)
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                v_pred = model(x_t, t, clip_vec, qwen_emb)
                loss = criterion(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止 MoE 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            optimizer.step()
            warmup_scheduler.step()  # 更新学习率
            ema.update()  # 更新 EMA 模型

            global_step += 1

            # --- 🔥 5. WandB 监控专家负载 ---
            if global_step % cfg.train.log_interval == 0:
                expert_counts = torch.zeros(cfg.model.num_experts, device=device)

                # 遍历模型里所有的 MoE 层，收集 last_indices
                for _name, module in model.named_modules():
                    if hasattr(module, "last_indices"):
                        # indices: [B, L, 2] -> flatten -> 统计每个 ID 出现次数
                        idx = module.last_indices.flatten()
                        counts = torch.bincount(idx, minlength=cfg.model.num_experts)
                        expert_counts += counts.float()

                # 归一化，看百分比
                expert_dist = expert_counts / expert_counts.sum()

                # 记录到 WandB
                log_dict = {"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "epoch": epoch}
                # 记录每个专家的忙碌程度
                for i in range(cfg.model.num_experts):
                    log_dict[f"expert_{i}_load"] = expert_dist[i].item()

                wandb.log(log_dict)
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            # --- 🔥 6. 训练中验证 (Validation Strategy) ---
            if global_step % cfg.train.validation_interval == 0:
                logger.info("🎨 Running validation inference...")

                # 保存临时权重
                ckpt_path = os.path.join(cfg.CKPT_DIR, "temp_val.pth")

                # 💡 关键：我们用 EMA 模型的权重来生成，因为它的画质更好
                # 如果 EMA 还没准备好 (step < 100)，就用普通 model
                save_model = ema.ema_model if global_step > 100 else model
                torch.save(save_model.state_dict(), ckpt_path)

                # 定义验证 Prompt
                val_prompt = "A high quality photo of a cyberpunk street, neon lights, night, 8k"
                out_path = os.path.join(cfg.CKPT_DIR, f"val_step_{global_step}.png")

                # 调用推理函数
                try:
                    run_inference(
                        ckpt_path=ckpt_path,
                        prompt=val_prompt,
                        output_path=out_path,
                        steps=20,  # 验证时步数少一点，为了快
                        width=512,
                        height=512,
                        seed=42,  # 固定种子，方便对比进化过程
                    )

                    # 把图传到 WandB
                    wandb.log({"validation_image": wandb.Image(out_path, caption=f"Step {global_step}")})
                except Exception as e:
                    logger.error(f"Validation failed: {e}")

                # 恢复训练模式
                model.train()

        # 每个 Epoch 保存一次权重
        if (epoch + 1) % cfg.train.save_interval == 0:
            save_path = os.path.join(cfg.CKPT_DIR, f"moe_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)

            # 同时也保存 EMA 权重 (通常这是我们最终想要发布的权重)
            ema_path = os.path.join(cfg.CKPT_DIR, f"moe_ep{epoch + 1}_ema.pth")
            torch.save(ema.ema_model.state_dict(), ema_path)

            logger.info(f"Saved checkpoint and EMA: {save_path}")

    wandb.finish()


if __name__ == "__main__":
    run_training()
