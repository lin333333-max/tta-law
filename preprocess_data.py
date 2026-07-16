"""
预处理数据：提前tokenize所有样本，缓存到磁盘
运行一次后，训练时直接加载缓存，避免实时分词
"""
import os
import json
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from config import Config

def preprocess_and_cache(config, data_file, output_file):
    """预处理数据并缓存"""
    print(f"\n{'='*60}")
    print(f"预处理数据: {data_file}")
    print(f"{'='*60}")

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    # 加载标签映射
    with open(config.label_mappings_file, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
        article_to_idx = {int(k): v for k, v in mappings['article_to_idx'].items()}
        accusation_to_idx = mappings['accusation_to_idx']

    # 加载原始数据
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    print(f"加载了 {len(raw_data)} 条样本")
    print(f"开始tokenize...")

    # 预处理所有样本
    processed_data = []
    for sample in tqdm(raw_data, desc="Tokenizing"):
        fact = sample['fact']

        # Tokenize
        encoding = tokenizer(
            fact,
            max_length=config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 法条标签
        articles = torch.zeros(config.num_articles, dtype=torch.float32)
        if 'meta' in sample and 'relevant_articles' in sample['meta']:
            for article in sample['meta']['relevant_articles']:
                article = int(article) if isinstance(article, str) else article
                if article in article_to_idx:
                    article_idx = article_to_idx[article]
                    if 0 <= article_idx < config.num_articles:
                        articles[article_idx] = 1.0

        # 罪名标签
        charges = torch.zeros(config.num_charges, dtype=torch.float32)
        if 'meta' in sample and 'accusation' in sample['meta']:
            for accusation in sample['meta']['accusation']:
                if accusation in accusation_to_idx:
                    charge_idx = accusation_to_idx[accusation]
                    if 0 <= charge_idx < config.num_charges:
                        charges[charge_idx] = 1.0

        # 保存处理后的数据
        processed_sample = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'articles': articles,
            'charges': charges,
            'fact_text': fact  # 保留原始文本
        }

        processed_data.append(processed_sample)

    # 保存到磁盘
    print(f"\n保存缓存到: {output_file}")
    torch.save(processed_data, output_file)

    # 显示文件大小
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ 缓存文件大小: {file_size:.1f} MB")
    print(f"✓ 处理完成: {len(processed_data)} 条样本\n")

def main():
    config = Config()
    cache_dir = os.path.join(config.data_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # 预处理训练集
    train_file = os.path.join(config.data_dir, config.train_file)
    train_cache = os.path.join(cache_dir, 'train_cached.pt')
    preprocess_and_cache(config, train_file, train_cache)

    # 预处理验证集
    dev_file = os.path.join(config.data_dir, config.dev_file)
    dev_cache = os.path.join(cache_dir, 'dev_cached.pt')
    preprocess_and_cache(config, dev_file, dev_cache)

    # 预处理测试集
    test_file = os.path.join(config.data_dir, config.test_file)
    test_cache = os.path.join(cache_dir, 'test_cached.pt')
    preprocess_and_cache(config, test_file, test_cache)

    print("="*60)
    print("✨ 所有数据预处理完成！")
    print("="*60)
    print("\n现在可以使用 FastLegalDataset 进行训练")
    print("训练速度预计提升 5-10 倍！\n")

if __name__ == '__main__':
    main()
