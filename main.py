import argparse
import os
import sys

# 1. 设置路径，确保能 import src 下的包
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from qfm.config import cfg  # noqa: E402
from qfm.core.logger import logger  # noqa: E402


def get_latest_checkpoint():
    """自动寻找最新权重的辅助函数 (.ckpt 或 .pth)"""
    if not os.path.exists(cfg.CKPT_DIR):
        return None
    # 优先找 lightning 的 .ckpt
    files = [f for f in os.listdir(cfg.CKPT_DIR) if f.endswith(".ckpt") or f.endswith(".pth")]
    if not files:
        return None
    # 按时间倒序
    files.sort(key=lambda x: os.path.getmtime(os.path.join(cfg.CKPT_DIR, x)))
    return os.path.join(cfg.CKPT_DIR, files[-1])


def main():
    parser = argparse.ArgumentParser(description="QFM Manager")
    parser.add_argument("mode", choices=["preprocess", "train", "inference"], help="Task to run")

    # Inference Args
    parser.add_argument("--prompt", type=str, default="A cyberpunk city with neon lights", help="Prompt")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="result.png")

    args = parser.parse_args()

    # ================= PREPROCESS =================
    if args.mode == "preprocess":
        from qfm.engine.preprocess import main as run_preprocess

        logger.info("🛠️  Starting Preprocessing...")
        run_preprocess()

    # ================= TRAIN (Lightning) =================
    elif args.mode == "train":
        from qfm.engine.trainer import run_training

        logger.info("🏋️  Starting Training (Lightning)...")
        run_training()

    # ================= INFERENCE =================
    elif args.mode == "inference":
        from qfm.engine.inference import run_inference

        ckpt_path = args.ckpt or get_latest_checkpoint()
        if not ckpt_path:
            logger.error("❌ No checkpoint found. Train first or specify --ckpt.")
            return

        logger.info(f"🎨 Inference with: {ckpt_path}")
        run_inference(
            ckpt_path=ckpt_path,
            prompt=args.prompt,
            output_path=args.output,
            steps=args.steps,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
