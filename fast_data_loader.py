"""
快速数据加载器：使用预处理缓存的数据
"""
import torch
from torch.utils.data import Dataset, DataLoader
import os

class FastLegalDataset(Dataset):
    """使用预处理缓存的快速数据集"""

    def __init__(self, cache_file: str):
        """
        Args:
            cache_file: 缓存文件路径（.pt格式）
        """
        print(f"Loading cached data from {cache_file}...")
        self.data = torch.load(cache_file)
        print(f"✓ Loaded {len(self.data)} preprocessed samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接返回预处理好的数据，无需实时tokenize
        return self.data[idx]


def get_fast_dataloader(config, cache_file: str, shuffle: bool = True) -> DataLoader:
    """创建快速DataLoader（使用缓存数据）"""
    dataset = FastLegalDataset(cache_file)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size if shuffle else config.eval_batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    return dataloader
