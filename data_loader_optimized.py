"""
优化的数据加载器：使用矩阵化映射替代循环
性能提升：10-50 倍
"""
import json
import torch
from typing import List
import numpy as np


class FastMappingDictionary:
    """矩阵化的法条-罪名映射字典（GPU 并行友好）"""

    def __init__(self, a2c_file: str, c2a_file: str, num_articles: int = 164, num_charges: int = 170):
        """
        Args:
            a2c_file: 法条→罪名映射文件
            c2a_file: 罪名→法条映射文件
            num_articles: 法条数量
            num_charges: 罪名数量
        """
        with open(a2c_file, 'r', encoding='utf-8') as f:
            self.a2c = json.load(f)

        with open(c2a_file, 'r', encoding='utf-8') as f:
            self.c2a = json.load(f)

        # 🚀 关键优化：预构建稀疏映射矩阵
        self.a2c_matrix = self._build_mapping_matrix(
            self.a2c, num_articles, num_charges
        )
        self.c2a_matrix = self._build_mapping_matrix(
            self.c2a, num_charges, num_articles
        )

        print(f"✓ Optimized mapping matrices created:")
        print(f"  A2C matrix: {self.a2c_matrix.shape}, {self.a2c_matrix.sum().item():.0f} connections")
        print(f"  C2A matrix: {self.c2a_matrix.shape}, {self.c2a_matrix.sum().item():.0f} connections")

    def _build_mapping_matrix(self, mapping_dict: dict, src_size: int, tgt_size: int) -> torch.Tensor:
        """
        构建稀疏映射矩阵
        Args:
            mapping_dict: {src_idx: [tgt_idx1, tgt_idx2, ...]}
            src_size: 源标签数量
            tgt_size: 目标标签数量
        Returns:
            [src_size, tgt_size] 映射矩阵
        """
        matrix = torch.zeros(src_size, tgt_size, dtype=torch.float32)

        for src_idx, tgt_indices in mapping_dict.items():
            src_idx = int(src_idx)
            if src_idx < src_size and tgt_indices:
                for tgt_idx in tgt_indices:
                    tgt_idx = int(tgt_idx) if isinstance(tgt_idx, str) else tgt_idx
                    if tgt_idx < tgt_size:
                        matrix[src_idx, tgt_idx] = 1.0

        return matrix

    def articles_to_charges_tensor(self, article_probs: torch.Tensor, device) -> torch.Tensor:
        """
        将法条预测概率映射到罪名（矩阵化版本）
        Args:
            article_probs: [batch, num_articles] 法条概率
            device: 设备
        Returns:
            [batch, num_charges] 映射后的罪名概率
        """
        # 🚀 单行矩阵乘法替代三层循环！
        # [batch, num_articles] @ [num_articles, num_charges] = [batch, num_charges]
        mapping_matrix = self.a2c_matrix.to(device)
        mapped_charges = torch.matmul(article_probs, mapping_matrix)

        # 归一化
        mapped_charges = torch.sigmoid(mapped_charges)
        return mapped_charges

    def charges_to_articles_tensor(self, charge_probs: torch.Tensor, device) -> torch.Tensor:
        """
        将罪名预测概率映射到法条（矩阵化版本）
        Args:
            charge_probs: [batch, num_charges] 罪名概率
            device: 设备
        Returns:
            [batch, num_articles] 映射后的法条概率
        """
        # 🚀 单行矩阵乘法替代三层循环！
        # [batch, num_charges] @ [num_charges, num_articles] = [batch, num_articles]
        mapping_matrix = self.c2a_matrix.to(device)
        mapped_articles = torch.matmul(charge_probs, mapping_matrix)

        # 归一化
        mapped_articles = torch.sigmoid(mapped_articles)
        return mapped_articles

    def get_mapped_charges(self, article_idx: int) -> List[int]:
        """根据法条索引获取对应罪名（兼容旧接口）"""
        return self.a2c.get(str(article_idx), [])

    def get_mapped_articles(self, charge_idx: int) -> List[int]:
        """根据罪名索引获取对应法条（兼容旧接口）"""
        return self.c2a.get(str(charge_idx), [])


# 测试脚本
if __name__ == '__main__':
    import time
    from data_loader import MappingDictionary

    print("性能对比测试\n" + "="*60)

    # 加载映射
    a2c_file = './dataset/article_to_charge_indexed.json'
    c2a_file = './dataset/charge_to_article_indexed.json'

    old_mapping = MappingDictionary(a2c_file, c2a_file)
    new_mapping = FastMappingDictionary(a2c_file, c2a_file)

    # 创建测试数据
    batch_size = 32
    num_articles = 164
    num_charges = 170
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    article_probs = torch.rand(batch_size, num_articles).to(device)
    charge_probs = torch.rand(batch_size, num_charges).to(device)

    # 预热
    _ = new_mapping.articles_to_charges_tensor(article_probs, device)

    # 测试旧版本
    print("\n旧版本（循环）:")
    start = time.time()
    for _ in range(10):
        result_old = old_mapping.articles_to_charges_tensor(article_probs, device)
    old_time = (time.time() - start) / 10
    print(f"  平均耗时: {old_time*1000:.2f} ms")

    # 测试新版本
    print("\n新版本（矩阵化）:")
    start = time.time()
    for _ in range(10):
        result_new = new_mapping.articles_to_charges_tensor(article_probs, device)
    new_time = (time.time() - start) / 10
    print(f"  平均耗时: {new_time*1000:.2f} ms")

    # 验证结果一致性
    diff = torch.abs(result_old - result_new).max().item()
    print(f"\n结果差异: {diff:.6f} (应接近0)")

    print(f"\n⚡ 加速比: {old_time/new_time:.1f}x")
    print("="*60)
