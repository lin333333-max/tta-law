"""
数据处理模块：加载和处理法律判决预测数据
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, List, Tuple
import numpy as np


class LegalDataset(Dataset):
    """法律判决预测数据集"""

    def __init__(self, data_file: str, tokenizer, max_length: int,
                 num_articles: int, num_charges: int, label_mappings_file: str = './dataset/label_mappings.json'):
        """
        Args:
            data_file: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            num_articles: 法条总数
            num_charges: 罪名总数
            label_mappings_file: 标签映射文件路径
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_articles = num_articles
        self.num_charges = num_charges

        # 加载标签映射
        with open(label_mappings_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.article_to_idx = {int(k): v for k, v in mappings['article_to_idx'].items()}
            self.accusation_to_idx = mappings['accusation_to_idx']

        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

        print(f"Loaded {len(self.data)} samples from {data_file}")
        print(f"Loaded label mappings: {num_articles} articles, {num_charges} charges")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 事实描述
        fact = sample['fact']

        # Tokenize
        encoding = self.tokenizer(
            fact,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 法条标签（多标签）
        articles = torch.zeros(self.num_articles, dtype=torch.float32)
        if 'meta' in sample and 'relevant_articles' in sample['meta']:
            for article in sample['meta']['relevant_articles']:
                # 法条是数字，转换为索引
                article = int(article) if isinstance(article, str) else article
                if article in self.article_to_idx:
                    article_idx = self.article_to_idx[article]
                    if 0 <= article_idx < self.num_articles:
                        articles[article_idx] = 1.0

        # 罪名标签（多标签）
        charges = torch.zeros(self.num_charges, dtype=torch.float32)
        if 'meta' in sample and 'accusation' in sample['meta']:
            for accusation in sample['meta']['accusation']:
                # 罪名是文本，转换为索引
                if accusation in self.accusation_to_idx:
                    charge_idx = self.accusation_to_idx[accusation]
                    if 0 <= charge_idx < self.num_charges:
                        charges[charge_idx] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'articles': articles,
            'charges': charges,
            'fact_text': fact  # 保留原始文本用于语义扰动
        }


class MappingDictionary:
    """法条-罪名映射字典"""

    def __init__(self, a2c_file: str, c2a_file: str):
        """
        Args:
            a2c_file: 法条→罪名映射文件
            c2a_file: 罪名→法条映射文件
        """
        with open(a2c_file, 'r', encoding='utf-8') as f:
            self.a2c = json.load(f)  # {article_idx: [charge_idx1, charge_idx2, ...]}

        with open(c2a_file, 'r', encoding='utf-8') as f:
            self.c2a = json.load(f)  # {charge_idx: [article_idx1, article_idx2, ...]}

        print(f"Loaded mapping dictionaries: {len(self.a2c)} articles, {len(self.c2a)} charges")

    def get_mapped_charges(self, article_idx: int) -> List[int]:
        """根据法条索引获取对应罪名"""
        return self.a2c.get(str(article_idx), [])

    def get_mapped_articles(self, charge_idx: int) -> List[int]:
        """根据罪名索引获取对应法条"""
        return self.c2a.get(str(charge_idx), [])

    def articles_to_charges_tensor(self, article_probs: torch.Tensor, device) -> torch.Tensor:
        """
        将法条预测概率映射到罪名
        Args:
            article_probs: [batch, num_articles] 法条概率
            device: 设备
        Returns:
            [batch, num_charges] 映射后的罪名概率
        """
        batch_size, num_articles = article_probs.shape
        # 修复：转换字符串索引为整数
        num_charges = max([max([int(c) for c in charges]) for charges in self.a2c.values() if charges]) + 1

        mapped_charges = torch.zeros(batch_size, num_charges, device=device)

        for batch_idx in range(batch_size):
            for article_idx in range(num_articles):
                charge_indices = self.get_mapped_charges(article_idx)
                if charge_indices and article_probs[batch_idx, article_idx] > 0:
                    for charge_idx in charge_indices:
                        # 转换为整数（处理字符串类型）
                        charge_idx = int(charge_idx) if isinstance(charge_idx, str) else charge_idx
                        if charge_idx < num_charges:
                            # 累积映射概率
                            mapped_charges[batch_idx, charge_idx] += article_probs[batch_idx, article_idx]

        # 归一化
        mapped_charges = torch.sigmoid(mapped_charges)
        return mapped_charges

    def charges_to_articles_tensor(self, charge_probs: torch.Tensor, device) -> torch.Tensor:
        """
        将罪名预测概率映射到法条
        Args:
            charge_probs: [batch, num_charges] 罪名概率
            device: 设备
        Returns:
            [batch, num_articles] 映射后的法条概率
        """
        batch_size, num_charges = charge_probs.shape
        # 修复：转换字符串索引为整数
        num_articles = max([max([int(a) for a in articles]) for articles in self.c2a.values() if articles]) + 1

        mapped_articles = torch.zeros(batch_size, num_articles, device=device)

        for batch_idx in range(batch_size):
            for charge_idx in range(num_charges):
                article_indices = self.get_mapped_articles(charge_idx)
                if article_indices and charge_probs[batch_idx, charge_idx] > 0:
                    for article_idx in article_indices:
                        # 转换为整数（处理字符串类型）
                        article_idx = int(article_idx) if isinstance(article_idx, str) else article_idx
                        if article_idx < num_articles:
                            # 累积映射概率
                            mapped_articles[batch_idx, article_idx] += charge_probs[batch_idx, charge_idx]

        # 归一化
        mapped_articles = torch.sigmoid(mapped_articles)
        return mapped_articles


def create_mapping_dict_from_cail(train_file: str, output_a2c: str, output_c2a: str):
    """
    从CAIL数据集构建映射字典
    这是一个辅助函数，用于首次构建映射字典
    """
    from collections import defaultdict

    a2c_dict = defaultdict(set)
    c2a_dict = defaultdict(set)

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if 'meta' not in sample:
                continue

            articles = sample['meta'].get('relevant_articles', [])
            charges = sample['meta'].get('accusation', [])

            # 构建双向映射
            for article in articles:
                for charge in charges:
                    a2c_dict[article].add(charge)
                    c2a_dict[charge].add(article)

    # 转换为列表
    a2c_dict = {k: list(v) for k, v in a2c_dict.items()}
    c2a_dict = {k: list(v) for k, v in c2a_dict.items()}

    # 保存
    with open(output_a2c, 'w', encoding='utf-8') as f:
        json.dump(a2c_dict, f, ensure_ascii=False, indent=2)

    with open(output_c2a, 'w', encoding='utf-8') as f:
        json.dump(c2a_dict, f, ensure_ascii=False, indent=2)

    print(f"Created mapping dictionaries: {len(a2c_dict)} articles, {len(c2a_dict)} charges")


def get_dataloader(config, data_file: str, tokenizer, shuffle: bool = True) -> DataLoader:
    """创建DataLoader"""
    dataset = LegalDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        num_articles=config.num_articles,
        num_charges=config.num_charges,
        label_mappings_file=config.label_mappings_file
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size if shuffle else config.eval_batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == "__main__":
    # 示例：构建映射字典
    # create_mapping_dict_from_cail(
    #     './dataset/train.json',
    #     './dataset/article_to_charge.json',
    #     './dataset/charge_to_article.json'
    # )
    pass
