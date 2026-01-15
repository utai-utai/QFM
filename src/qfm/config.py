import os
from dataclasses import dataclass

import torch


@dataclass
class ModelArgs:
    input_size: int = 64
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1024
    depth: int = 8
    num_heads: int = 16
    num_experts: int = 8
    qwen_dim: int = 1536
    clip_dim: int = 768


@dataclass
class TrainingArgs:
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 10
    save_interval: int = 5
    grad_clip: float = 1.0
    seed: int = 42


class Config:
    # 1. 获取根目录
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(_current_dir))

    # 2. 路径配置
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    JSONL_PATH = os.path.join(DATA_DIR, "data.jsonl")
    CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

    # Model IDs
    VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
    CLIP_ID = "openai/clip-vit-large-patch14"
    LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    model = ModelArgs()
    train = TrainingArgs()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_dtype(device):
        if device.type == "cuda":
            return torch.bfloat16
        # elif device.type == "mps":
        #     return torch.float16
        return torch.float32


cfg = Config()
