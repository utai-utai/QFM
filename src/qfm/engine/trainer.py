import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from qfm.config import cfg
from qfm.core.logger import logger
from qfm.dataset import LatentDataset
from qfm.model_moe import MiniFluxDiT


def run_training():
    device = cfg.get_device()
    dtype = cfg.get_dtype(device)

    logger.info(f"🚀 Starting Training on {device} ({dtype})")
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    # Init Model
    model = MiniFluxDiT(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        num_experts=cfg.model.num_experts,
        qwen_dim=cfg.model.qwen_dim,
        clip_dim=cfg.model.clip_dim,
    ).to(device, dtype=dtype)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # Dataset
    try:
        dataset = LatentDataset(cfg.PROCESSED_DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(cfg.train.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        loss_avg = 0

        for step, batch in enumerate(pbar):
            x_1 = batch["x"].to(device, dtype=dtype)
            clip_vec = batch["t_embed"].to(device, dtype=dtype)
            qwen_emb = batch["context"].to(device, dtype=dtype)

            batch_size = x_1.shape[0]

            # Flow Matching Logic
            t = torch.rand(batch_size, device=device, dtype=dtype)
            x_0 = torch.randn_like(x_1)
            t_expand = t.view(batch_size, 1, 1, 1)
            x_t = t_expand * x_1 + (1 - t_expand) * x_0
            v_target = x_1 - x_0

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                v_pred = model(x_t, t, clip_vec, qwen_emb)
                loss = criterion(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            loss_avg = loss_avg * 0.9 + loss.item() * 0.1
            pbar.set_postfix(loss=f"{loss_avg:.4f}")

        if (epoch + 1) % cfg.train.save_interval == 0:
            save_path = os.path.join(cfg.CKPT_DIR, f"qfm_moe_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint: {save_path}")
