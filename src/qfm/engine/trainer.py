import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from qfm.config import cfg
from qfm.core.logger import logger
from qfm.dataset import BucketedBatchSampler, LatentDataset
from qfm.model_moe import MiniFluxDiT


def run_training():
    # 1. 硬件加速优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = cfg.get_device()
    dtype = cfg.get_dtype(device)

    logger.info(f"🚀 Starting Training on {device} ({dtype})")
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    # 2. 初始化模型
    model = MiniFluxDiT(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        num_experts=cfg.model.num_experts,
        qwen_dim=cfg.model.qwen_dim,
        clip_dim=cfg.model.clip_dim,
    ).to(device, dtype=dtype)

    # 💡 如果配置了梯度检查点，尝试开启 (需要在 Model 里实现该方法，或者直接在 forward 里用)
    if hasattr(cfg.train, "gradient_checkpointing") and cfg.train.gradient_checkpointing:
        logger.info("🛡️ Gradient Checkpointing Enabled (Saving VRAM)")
        # 假设你在 model_moe.py 里实现了这个方法，如果没有，请确保你的 forward 支持
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            logger.warning(
                "⚠️ Config requested gradient checkpointing but model doesn't "
                "have 'gradient_checkpointing_enable' method."
            )

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # 3. 数据加载 (核心修改: 使用 BucketedBatchSampler)
    try:
        dataset = LatentDataset(cfg.PROCESSED_DATA_DIR)

        # 🔥 初始化分桶采样器
        sampler = BucketedBatchSampler(dataset, batch_size=cfg.train.batch_size, shuffle=True)

        # 🔥 DataLoader 变更:
        # - batch_sampler=sampler: 接管 batch_size 和 shuffle
        # - shuffle=False: 必须关掉，否则冲突
        # - drop_last=False: Sampler 也会接管
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=4,  # WSL2/Linux 推荐 4-8
            pin_memory=True,  # 锁页内存，加快 CPU -> GPU 传输
        )
        logger.info(f"Data Loaded: {len(dataset)} samples in {len(dataloader)} batches (Bucketed).")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        # 打印堆栈以便调试
        import traceback

        traceback.print_exc()
        return

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.train()

    # 4. 训练循环
    for epoch in range(cfg.train.epochs):
        # sampler.set_epoch(epoch) # 如果 BucketedBatchSampler 内部实现了 determinism 需要这行，目前你的简易版不需要

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        loss_avg = 0

        # 记录 Step 数，方便 Save Interval
        global_step = epoch * len(dataloader)

        for _step, batch in enumerate(pbar):
            # 这里的 batch["x"] 形状会根据桶自动变化，例如 [16, 4, 64, 64] 或 [16, 4, 48, 80]
            x_1 = batch["x"].to(device, dtype=dtype)
            clip_vec = batch["t_embed"].to(device, dtype=dtype)
            qwen_emb = batch["context"].to(device, dtype=dtype)

            batch_size = x_1.shape[0]

            # Flow Matching Logic
            t = torch.rand(batch_size, device=device, dtype=dtype)
            x_0 = torch.randn_like(x_1)
            t_expand = t.view(batch_size, 1, 1, 1)

            # Straight flow: x_t = t * x_1 + (1-t) * x_0
            x_t = t_expand * x_1 + (1 - t_expand) * x_0
            v_target = x_1 - x_0

            # 混合精度上下文
            # 注意: 如果 dtype 已经是 bfloat16，autocast 其实是可选的，但加上比较保险
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                v_pred = model(x_t, t, clip_vec, qwen_emb)
                loss = criterion(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            loss_avg = loss_avg * 0.9 + loss.item() * 0.1
            pbar.set_postfix(loss=f"{loss_avg:.4f}", shape=f"{x_1.shape[2]}x{x_1.shape[3]}")

            global_step += 1

        # Save Checkpoint
        if (epoch + 1) % cfg.train.save_interval == 0:
            save_path = os.path.join(cfg.CKPT_DIR, f"qfm_moe_ep{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                save_path,
            )
            logger.info(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    run_training()
