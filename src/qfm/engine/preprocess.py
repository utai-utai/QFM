import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from qfm.config import cfg

# =================配置区域=================
# 定义分桶的分辨率 (Width, Height)
# 保证总像素数约为 256k (512x512) 或 1M (1024x1024)
# 这里以 512为基准，适配 3090 Ti 显存
BUCKETS = [
    (512, 512),  # 1:1
    (576, 448),  # 4:3 这里的数字必须能被 8 (VAE stride) 整除，最好能被 32 (DiT patch) 整除
    (448, 576),  # 3:4
    (640, 384),  # 16:9
    (384, 640),  # 9:16
    (704, 320),  # 21:9
    (320, 704),  # 9:21
]


def get_closest_bucket(image):
    """计算图片最接近哪个桶"""
    w, h = image.size
    aspect_ratio = w / h

    best_bucket = None
    min_diff = float("inf")

    for bw, bh in BUCKETS:
        bucket_ratio = bw / bh
        diff = abs(aspect_ratio - bucket_ratio)
        if diff < min_diff:
            min_diff = diff
            best_bucket = (bw, bh)

    return best_bucket


def resize_and_crop(image, target_res):
    """将图片缩放并中心裁剪到目标桶大小"""
    target_w, target_h = target_res

    # 1. Resize (保持比例)
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 2. Center Crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    image = image.crop((left, top, right, bottom))
    return image


@torch.no_grad()
def main():
    # 1. 准备环境
    device = cfg.get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"🚀 Starting Preprocess on {device} with {dtype}")

    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)

    # 2. 加载模型 (VAE + CLIP + Qwen)
    print("⏳ Loading Models...")

    # VAE
    vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device, dtype)

    # CLIP
    clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.CLIP_ID)
    clip_encoder = CLIPTextModel.from_pretrained(cfg.CLIP_ID).to(device, dtype)

    # Qwen (LLM)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_ID)
    qwen_model = AutoModelForCausalLM.from_pretrained(cfg.LLM_ID, torch_dtype=dtype).to(device)

    print("✅ Models Loaded.")

    # 3. 读取数据索引
    if not os.path.exists(cfg.JSONL_PATH):
        raise FileNotFoundError(f"❌ Cannot find {cfg.JSONL_PATH}")

    with open(cfg.JSONL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"📊 Found {len(lines)} images. Processing...")

    # 用于保存处理后的索引 (包含 buckets 信息)
    processed_index = []

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        try:
            item = json.loads(line)
            rel_path = item["image"]  # data/raw_images/xxx.jpg
            prompt = item["text"]

            # 构建绝对路径
            # 注意：如果你的 data 目录里做了软链接，这里 os.path.join 依然有效
            full_img_path = os.path.join(cfg.PROJECT_ROOT, rel_path)

            if not os.path.exists(full_img_path):
                # 尝试修复路径：如果 jsonl 里写的是绝对路径或不匹配
                # 简单处理：假设文件名是对的，去 RAW_IMAGES_DIR 找
                filename = os.path.basename(rel_path)
                full_img_path = os.path.join(cfg.RAW_IMAGES_DIR, filename)
                if not os.path.exists(full_img_path):
                    print(f"⚠️ Skip: Image not found {rel_path}")
                    continue

            # --- Image Processing (Bucketing) ---
            image = Image.open(full_img_path).convert("RGB")

            # 找桶 + 裁剪
            bucket_w, bucket_h = get_closest_bucket(image)
            image = resize_and_crop(image, (bucket_w, bucket_h))

            # 归一化 [-1, 1]
            img_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype)  # [1, 3, H, W]

            # VAE Encode
            # 乘以 Scaling Factor 0.13025 (SDXL标准)
            latents = vae.encode(img_tensor).latent_dist.sample() * 0.13025

            # --- Text Processing ---

            # 1. CLIP Embedding (Pooling output)
            clip_inputs = clip_tokenizer(
                prompt, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            clip_out = clip_encoder(**clip_inputs)
            clip_embeds = clip_out.pooler_output  # [1, 768] (ModelArgs.clip_dim)

            # 2. Qwen Embedding (Hidden states)
            qwen_inputs = qwen_tokenizer(
                prompt, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                qwen_out = qwen_model(**qwen_inputs, output_hidden_states=True)
                # 取最后一层 hidden state
                qwen_embeds = qwen_out.hidden_states[-1]  # [1, 128, 1536] (ModelArgs.qwen_dim)

            # --- Save to Disk ---
            save_name = f"{idx:08d}.pt"
            save_path = os.path.join(cfg.PROCESSED_DATA_DIR, save_name)

            torch.save(
                {
                    "vae_latent": latents.squeeze(0).cpu(),  # [4, H/8, W/8]
                    "clip_embeds": clip_embeds.squeeze(0).cpu(),  # [768]
                    "qwen_embeds": qwen_embeds.squeeze(0).cpu(),  # [128, 1536]
                    "bucket_res": (bucket_h, bucket_w),  # 记录分辨率，DataLoader要用！
                },
                save_path,
            )

            # 添加到新索引
            processed_index.append(
                {
                    "file": save_name,
                    "shape": (bucket_h // 8, bucket_w // 8),  # 存 Latent 的形状
                }
            )

        except Exception as e:
            print(f"❌ Error processing {idx}: {e}")
            continue

    # 保存新的索引文件，包含分辨率信息
    processed_json_path = os.path.join(cfg.PROCESSED_DATA_DIR, "processed_index.json")
    with open(processed_json_path, "w") as f:
        json.dump(processed_index, f)

    print(f"✅ Done! Processed {len(processed_index)} images.")
    print(f"📄 Index saved to {processed_json_path}")


if __name__ == "__main__":
    main()
