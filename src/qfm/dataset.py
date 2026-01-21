import json
import os
import random

import torch
from torch.utils.data import Dataset, Sampler

from qfm.config import cfg


class LatentDataset(Dataset):
    def __init__(self, data_dir=None):
        """
        data_dir: 实际上现在主要依赖 cfg.PROCESSED_INDEX_PATH
        """
        super().__init__()
        self.data_dir = data_dir or cfg.PROCESSED_DATA_DIR
        self.index_path = cfg.PROCESSED_INDEX_PATH

        if not os.path.exists(self.index_path):
            raise ValueError(f"❌ Index not found: {self.index_path}. Please run preprocess.py first.")

        # 加载索引 [{"file": "001.pt", "shape": [64, 64]}, ...]
        with open(self.index_path, "r") as f:
            self.metadata = json.load(f)

        print(f"Dataset: Loaded {len(self.metadata)} samples from index.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item_info = self.metadata[idx]
        filename = item_info["file"]
        path = os.path.join(self.data_dir, filename)

        try:
            # map_location="cpu" 防止多进程加载时 GPU 显存溢出
            data = torch.load(path, map_location="cpu", weights_only=False)

            # 🔥 关键修改：对齐 preprocess.py 的 Key，并转换成模型需要的 Key
            return {
                "x": data["vae_latent"].float(),  # [4, H, W] (变化的形状)
                "t_embed": data["clip_embeds"].float(),  # [768]
                "context": data["qwen_embeds"].float(),  # [Seq, 1536]
                # 还可以返回 resolution 用于调试，但模型不需要
            }
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


class BucketedBatchSampler(Sampler):
    """
    分桶采样器：
    1. 按照图片形状(shape)将索引分组。
    2. 在每个组内生成 Batch。
    3. 将所有生成的 Batch 打乱顺序输出。
    这样 DataLoader 拿到的每一个 Batch，内部的图片形状都是严格一致的。
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 1. 扫描所有数据，按 shape 分组
        # buckets 格式: { (64, 64): [0, 3, 5...], (64, 96): [1, 2, ...] }
        self.buckets = {}
        for idx, item in enumerate(dataset.metadata):
            # item['shape'] 是 list [h, w]，转成 tuple 才能做字典 key
            shape = tuple(item["shape"])
            if shape not in self.buckets:
                self.buckets[shape] = []
            self.buckets[shape].append(idx)

        self.batches = []
        self._create_batches()

    def _create_batches(self):
        self.batches = []
        for _shape, indices in self.buckets.items():
            # 如果需要 Shuffle，打乱该桶内的索引
            if self.shuffle:
                random.shuffle(indices)

            # 按 batch_size 切片
            # drop_last=False (即使最后不够一个 batch 也保留，避免浪费数据)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                self.batches.append(batch)

    def __iter__(self):
        if self.shuffle:
            self._create_batches()  # 重新打乱桶内顺序
            random.shuffle(self.batches)  # 打乱 Batch 之间的顺序（防止模型一直看横图，再一直看竖图）

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
