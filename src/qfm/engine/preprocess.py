import argparse
import json
import os

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from qfm.config import cfg
from qfm.core.logger import logger

MODEL_CONFIG = {
    "vae": "madebyollin/sdxl-vae-fp16-fix",
    "clip": "openai/clip-vit-large-patch14",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
}
VAE_SCALING_FACTOR = 0.13025


class ImageTextDataset(Dataset):
    def __init__(self, jsonl_path, image_size=512):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image"]).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros((3, 512, 512))
        return {"pixel_values": pixel_values, "text": item["text"], "id": f"{idx:05d}"}


def main():
    device = cfg.get_device()
    dtype = cfg.get_dtype(device)
    logger.info(f"device: {device}, dtype: {dtype}")
    logger.info("Loading models...")
    vae = AutoencoderKL.from_pretrained(MODEL_CONFIG["vae"]).to(device, dtype=dtype).eval()
    clip_tok = CLIPTokenizer.from_pretrained(MODEL_CONFIG["clip"])
    clip_model = CLIPTextModel.from_pretrained(MODEL_CONFIG["clip"]).to(device, dtype=dtype).eval()
    qwen_tok = AutoTokenizer.from_pretrained(MODEL_CONFIG["qwen"])
    qwen_model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["qwen"]).to(device, dtype=dtype).eval()

    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    dataset = ImageTextDataset(cfg.JSONL_PATH, cfg.model.input_size * 8)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader):
            px = batch["pixel_values"].to(device, dtype=dtype)
            txt = batch["text"]
            ids = batch["id"]

            # VAE
            latents = vae.encode(px).latent_dist.sample() * VAE_SCALING_FACTOR

            # CLIP
            c_in = clip_tok(txt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            clip_vec = clip_model(**c_in).pooler_output  # [B, 768]

            # Qwen
            q_in = qwen_tok(
                txt, padding="max_length", max_length=cfg.model.qwen_dim, truncation=True, return_tensors="pt"
            ).to(device)
            qwen_emb = qwen_model(**q_in, output_hidden_states=True).hidden_states[-1]  # [B, Seq, 1536]

            # Save
            for i in range(len(ids)):
                torch.save(
                    {
                        "vae_latent": latents[i].cpu().clone(),
                        "clip_vec": clip_vec[i].cpu().clone(),
                        "qwen_emb": qwen_emb[i].cpu().clone(),
                    },
                    os.path.join(cfg.PROCESSED_DATA_DIR, f"{ids[i]}.pt"),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    main(parser.parse_args())
