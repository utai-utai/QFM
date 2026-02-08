import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Sampler

# ==========================================
# 1. 导入项目组件
# ==========================================
from qfm.config import cfg
from qfm.core.logger import logger
from qfm.dataset import LatentDataset

# 导入推理函数用于 Validation
from qfm.engine.inference import run_inference
from qfm.model_moe import MiniFluxDiT, SparseMoELayer


# ==========================================
# 2. 分桶采样器 (兼容单卡/多卡)
# ==========================================
class BucketedBatchSampler(Sampler):
    """
    逻辑：全局分桶 -> Shuffle -> 生成 Batch -> (如果是多卡)按 Rank 切分
    """

    def __init__(self, dataset, batch_size, seed=42, shuffle=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        # 自动检测 DDP 状态
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1. 分桶
        buckets = {}
        for idx, item in enumerate(self.dataset.metadata):
            shape = tuple(item["shape"])
            if shape not in buckets:
                buckets[shape] = []
            buckets[shape].append(idx)

        # 2. 生成 Batches
        all_batches = []
        for _shape, idxs in buckets.items():
            if self.shuffle:
                idxs = torch.tensor(idxs)[torch.randperm(len(idxs), generator=g)].tolist()

            # 生成 Batch (单卡视角)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # 丢弃不完整的最后一行，保证 tensor 形状对齐
                    all_batches.append(batch)

        # 3. Global Shuffle
        if self.shuffle:
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]

        # 4. DDP 切分 (如果是单卡，slice 就是 [::1]，即全取)
        yield from all_batches[self.rank :: self.num_replicas]

    def __len__(self):
        return len(self.dataset) // (self.batch_size * self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch


# ==========================================
# 3. EMA Callback (复刻 ema_pytorch 逻辑)
# ==========================================
class EMACallback(Callback):
    def __init__(self, decay=0.9999, update_every=10, start_step=100):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.start_step = start_step
        self.ema_state_dict = {}

    def on_fit_start(self, trainer, pl_module):
        # 初始化影子权重
        self.ema_state_dict = {
            n: p.clone().detach().to(pl_module.device) for n, p in pl_module.named_parameters() if p.requires_grad
        }
        if trainer.is_global_zero:
            logger.info(f"✅ EMA initialized (decay={self.decay}, every={self.update_every} steps)")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1. 检查是否达到开始步数
        if trainer.global_step < self.start_step:
            return

        # 2. 检查是否达到更新间隔
        if trainer.global_step % self.update_every != 0:
            return

        # 3. 更新影子权重
        for n, p in pl_module.named_parameters():
            if p.requires_grad and n in self.ema_state_dict:
                self.ema_state_dict[n].mul_(self.decay).add_(p.data, alpha=(1.0 - self.decay))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存 checkpoint 时，把 EMA 权重也存进去
        checkpoint["ema_state_dict"] = self.ema_state_dict

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema_state_dict = checkpoint["ema_state_dict"]
            if trainer.is_global_zero:
                logger.info("✅ EMA state loaded from checkpoint.")


# ==========================================
# 4. 验证回调 (Validation Inference)
# ==========================================
class ImageLoggerCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 频率控制
        if trainer.global_step % cfg.train.validation_interval != 0 or trainer.global_step == 0:
            return

        # 只在主进程执行
        if not trainer.is_global_zero:
            return

        logger.info(f"🎨 [Step {trainer.global_step}] Running Validation...")

        # 1. 保存临时权重 (优先使用 EMA)
        ckpt_path = os.path.join(cfg.CKPT_DIR, "temp_val_weights.pth")

        # 获取 EMA Callback 实例
        ema_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, EMACallback):
                ema_cb = cb
                break

        state_dict_to_save = pl_module.model.state_dict()
        # 如果 EMA 存在且已经开始更新，尝试混合 EMA 权重
        if ema_cb and trainer.global_step > ema_cb.start_step:
            logger.info(f"✨ Using EMA weights for validation (Step {trainer.global_step})")

            # 1. 获取 EMA 字典
            ema_raw_dict = ema_cb.ema_state_dict

            # 2. 修正 Key 的前缀 (去除 "model.")
            # 因为 EMA callback 是从 pl_module 抓的参数，key 可能是 "model.x_embedder.weight"
            # 但推理时的 MiniFluxDiT 需要 "x_embedder.weight"
            clean_ema_dict = {}
            for k, v in ema_raw_dict.items():
                # 这里的 "model." 取决于你在 QFMModule 里把模型命名为 self.model
                if k.startswith("model."):
                    clean_k = k[6:]  # 去掉 "model." (6个字符)
                    clean_ema_dict[clean_k] = v
                else:
                    clean_ema_dict[k] = v

            # 3. 替换要保存的字典
            state_dict_to_save = clean_ema_dict

        torch.save(state_dict_to_save, ckpt_path)

        # 2. 定义参数
        val_prompt = "A high quality photo of a cyberpunk street, neon lights, night, 8k"
        out_path = os.path.join(cfg.CKPT_DIR, f"val_step_{trainer.global_step}.png")

        # 3. 调用推理
        try:
            run_inference(
                ckpt_path=ckpt_path,
                prompt=val_prompt,
                output_path=out_path,
                steps=20,  # 快速预览
                width=512,
                height=512,
                seed=42,
                device=str(pl_module.device),  # 传入当前 GPU
            )

            # 4. Log 到 WandB
            if trainer.logger:
                trainer.logger.log_image(
                    key="validation_image", images=[out_path], caption=[f"Step {trainer.global_step}"]
                )

        except Exception as e:
            logger.error(f"⚠️ Validation inference failed: {e}")

        # 切回训练模式
        pl_module.model.train()


# ==========================================
# 5. Lightning Module
# ==========================================
class QFMModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = MiniFluxDiT(
            input_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,  # 假设 config 有这个
            in_channels=cfg.model.in_channels,
            hidden_size=cfg.model.hidden_size,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            num_experts=cfg.model.num_experts,
            qwen_dim=cfg.model.qwen_dim,
            clip_dim=cfg.model.clip_dim,
        )

        # 缓存 MoE 层，用于计算 aux_loss
        self.moe_layers = [m for m in self.model.modules() if isinstance(m, SparseMoELayer)]

    def training_step(self, batch, batch_idx):
        # 1. 准备数据
        x_1 = batch["x"]
        clip_emb = batch["t_embed"]
        qwen_emb = batch["context"]

        batch_size = x_1.shape[0]

        # 2. Flow Matching: 构造 x_t 和 target
        t = torch.rand((batch_size,), device=self.device, dtype=x_1.dtype)
        x_0 = torch.randn_like(x_1)

        # t 广播
        t_expand = t.view(batch_size, 1, 1, 1)

        # 插值: x_t = t * x_1 + (1-t) * x_0
        x_t = t_expand * x_1 + (1 - t_expand) * x_0

        # 目标: v = x_1 - x_0
        v_target = x_1 - x_0

        # 3. 前向传播
        v_pred = self.model(x_t, t, clip_emb, qwen_emb)

        # 4. 主 Loss (MSE)
        mse_loss = F.mse_loss(v_pred, v_target)

        # 5. MoE 辅助 Loss (防止专家坍塌)
        # 即使你原来的 trainer 没加，加上这个绝对有益无害，权重给小点即可
        aux_loss = 0.0
        if self.moe_layers:
            aux_loss = sum(m.aux_loss for m in self.moe_layers if m.aux_loss is not None) / len(self.moe_layers)

        total_loss = mse_loss + 0.01 * aux_loss  # 0.01 权重很小，不影响主任务

        # 6. 日志
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/mse_loss", mse_loss)
        self.log("train/aux_loss", aux_loss)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        # 对应原来的 optim.AdamW
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)

        # 对应原来的 Warmup (LinearLR)
        # 前 2000 步从 0.01 倍 LR 线性增长到 1.0 倍
        scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=2000)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 每个 step 更新一次 LR
            },
        }

    def on_train_epoch_start(self):
        # 更新 Sampler 的 Epoch 以保证随机性
        if hasattr(self.trainer.datamodule, "sampler"):
            self.trainer.datamodule.sampler.set_epoch(self.current_epoch)


# ==========================================
# 6. DataModule
# ==========================================
class QFMDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        self.train_dataset = LatentDataset()

    def train_dataloader(self):
        self.sampler = BucketedBatchSampler(self.train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
        return DataLoader(self.train_dataset, batch_sampler=self.sampler, num_workers=4, pin_memory=True)


# ==========================================
# 7. 主运行入口
# ==========================================
def run_training():
    # 1. 硬件优化
    torch.set_float32_matmul_precision("high")

    # 2. 策略判断 (单卡 vs 多卡)
    strategy = "auto"
    devices = 1
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=True)  # MoE 必须 True
        devices = -1
        print("⚡ Multi-GPU Detected. Using DDP.")
    else:
        print("⚡ Single GPU/MPS Detected. Using Auto strategy.")

    # 3. 初始化组件
    dm = QFMDataModule()
    model = QFMModule()

    # WandB Logger
    wandb_logger = WandbLogger(
        project="QFM-Small-MoE", name="pl-run-v1", config={"batch_size": cfg.train.batch_size, "lr": cfg.train.lr}
    )

    # Callbacks
    callbacks = [
        # Checkpoint: 每隔 N epoch 保存
        ModelCheckpoint(
            dirpath=cfg.CKPT_DIR,
            filename="qfm-{epoch:02d}-{train/loss:.2f}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
            save_last=True,
            every_n_epochs=cfg.train.save_interval,
        ),
        # LR Monitor
        LearningRateMonitor(logging_interval="step"),
        # EMA (手动实现的版本)
        EMACallback(decay=0.9999, update_every=10, start_step=100),
        # Validation Inference
        ImageLoggerCallback(),
    ]

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed",  # 自动混合精度
        max_epochs=cfg.train.epochs,
        gradient_clip_val=cfg.train.grad_clip,  # 梯度裁剪
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_interval,
        num_sanity_val_steps=0,  # 跳过 Sanity Check 加速启动
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
    )

    # 5. 开始训练
    if trainer.is_global_zero:
        logger.info("🚀 Starting PyTorch Lightning Training...")

    trainer.fit(model, datamodule=dm)
