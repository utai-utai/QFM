import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))
from qfm.core.logger import logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Mini-Flux MVP Manager")
    parser.add_argument("mode", choices=["preprocess", "train", "inference"], help="Task to run")
    parser.add_argument("--prompt", type=str, help="Prompt for inference", default="A cyberpunk city")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path for inference")

    args = parser.parse_args()

    if args.mode == "preprocess":
        from qfm.engine.preprocess import main as run_preprocess

        # 你需要稍微修改 engine/preprocess.py 让它接受 args 或直接运行逻辑
        # 这里假设你把之前的 main 逻辑封装成了一个函数 run_preprocess()
        logger.info("Starting Preprocessing...")
        # 传递一个模拟的 args 对象或者直接修改 preprocess.py 不依赖命令行参数
        run_preprocess()

    elif args.mode == "train":
        from qfm.engine.trainer import run_training

        logger.info("Starting Training Engine...")
        run_training()

    elif args.mode == "inference":
        if not args.ckpt:
            logger.error("Please provide --ckpt for inference!")
            return
        from qfm.engine.inference import run_inference

        logger.info(f"Starting Inference with prompt: {args.prompt}")
        run_inference(args.ckpt, args.prompt, output_path="generated_result.png")


if __name__ == "__main__":
    main()
