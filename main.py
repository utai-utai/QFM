import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from qfm.config import cfg  # noqa: E402
from qfm.core.logger import logger  # noqa: E402


def get_latest_checkpoint():
    """辅助函数：自动寻找最新的权重文件"""
    if not os.path.exists(cfg.CKPT_DIR):
        return None
    ckpts = [f for f in os.listdir(cfg.CKPT_DIR) if f.endswith(".pth")]
    if not ckpts:
        return None
    # 按修改时间排序，找最新的
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(cfg.CKPT_DIR, x)))
    return os.path.join(cfg.CKPT_DIR, ckpts[-1])


def main():
    parser = argparse.ArgumentParser(description="Mini-Flux MVP Manager")

    # 核心任务选择
    parser.add_argument("mode", choices=["preprocess", "train", "inference"], help="Task to run")

    # Inference 专用参数
    parser.add_argument("--prompt", type=str, default="A cyberpunk city with neon lights", help="Prompt for inference")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path (optional, defaults to latest)")
    parser.add_argument("--width", type=int, default=512, help="Image Width (Inference only)")
    parser.add_argument("--height", type=int, default=512, help="Image Height (Inference only)")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="result.png", help="Output filename")

    args = parser.parse_args()

    # ================= PREPROCESS =================
    if args.mode == "preprocess":
        from qfm.engine.preprocess import main as run_preprocess

        logger.info("🛠️  Starting Preprocessing Pipeline...")
        run_preprocess()

    # ================= TRAIN =================
    elif args.mode == "train":
        from qfm.engine.trainer import run_training

        logger.info("🏋️  Starting Training Engine...")
        run_training()

    # ================= INFERENCE =================
    elif args.mode == "inference":
        from qfm.engine.inference import run_inference

        # 自动寻找权重逻辑
        ckpt_path = args.ckpt
        if not ckpt_path:
            logger.info("🔍 No checkpoint provided, looking for the latest one...")
            ckpt_path = get_latest_checkpoint()

        if not ckpt_path:
            logger.error(f"❌ No checkpoints found in {cfg.CKPT_DIR}. Please run training first or specify --ckpt.")
            return

        logger.info("🎨 Starting Inference...")
        logger.info(f"   Prompt: {args.prompt}")
        logger.info(f"   Size: {args.width}x{args.height}")

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
