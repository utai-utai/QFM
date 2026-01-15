import glob
import os

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.files:
            raise ValueError(f"No .pt files found in {data_dir}")
        print(f"Dataset: Found {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # Map location cpu prevents GPU OOM during loading
            data = torch.load(path, map_location="cpu", weights_only=False)
            return {
                "x": data["vae_latent"].float(),  # [4, 64, 64]
                "t_embed": data["clip_vec"].float(),  # [768]
                "context": data["qwen_emb"].float(),  # [Seq, 1536]
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
