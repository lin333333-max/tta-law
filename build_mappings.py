"""
构建罪名和法条的文本到索引映射
"""
import json
from collections import defaultdict

def build_label_mappings(train_file: str, output_dir: str = './dataset'):
    """
    从训练数据构建标签映射
    """
    print("="*60)
    print("构建标签映射")
    print("="*60)

    # 收集所有唯一的法条和罪名
    articles_set = set()
    accusations_set = set()

    # A2C和C2A映射
    article_to_charge = defaultdict(set)
    charge_to_article = defaultdict(set)

    print(f"\n读取训练数据: {train_file}")

    with open(train_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 10000 == 0:
                print(f"   处理进度: {idx} 样本...")

            data = json.loads(line)

            if 'meta' not in data:
                continue

            # 法条（数字）
            articles = data['meta'].get('relevant_articles', [])
            # 罪名（文本）
            accusations = data['meta'].get('accusation', [])

            # 收集唯一值
            for article in articles:
                articles_set.add(int(article))

            for accusation in accusations:
                accusations_set.add(accusation)

            # 构建双向映射
            for article in articles:
                for accusation in accusations:
                    article_to_charge[int(article)].add(accusation)
                    charge_to_article[accusation].add(int(article))

    print(f"\n数据统计:")
    print(f"   唯一法条数: {len(articles_set)}")
    print(f"   唯一罪名数: {len(accusations_set)}")

    # 创建法条索引映射（法条本身就是数字）
    article_list = sorted(list(articles_set))
    article_to_idx = {article: idx for idx, article in enumerate(article_list)}
    idx_to_article = {idx: article for article, idx in article_to_idx.items()}

    # 创建罪名索引映射（罪名是文本，需要映射到索引）
    accusation_list = sorted(list(accusations_set))
    accusation_to_idx = {acc: idx for idx, acc in enumerate(accusation_list)}
    idx_to_accusation = {idx: acc for acc, idx in accusation_to_idx.items()}

    print(f"\n法条范围: {min(article_list)} - {max(article_list)}")
    print(f"   前5个法条: {article_list[:5]}")

    print(f"\n罪名列表（前10个）:")
    for i, acc in enumerate(accusation_list[:10]):
        print(f"   {i}: {acc}")

    # 保存映射文件
    mappings = {
        'article_to_idx': article_to_idx,
        'idx_to_article': idx_to_article,
        'accusation_to_idx': accusation_to_idx,
        'idx_to_accusation': idx_to_accusation,
        'num_articles': len(article_list),
        'num_accusations': len(accusation_list)
    }

    label_mapping_file = f'{output_dir}/label_mappings.json'
    with open(label_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    print(f"\n标签映射已保存: {label_mapping_file}")

    # 保存A2C和C2A映射（使用索引）
    a2c_indexed = {}
    for article, charges in article_to_charge.items():
        article_idx = article_to_idx[article]
        charge_indices = [accusation_to_idx[c] for c in charges]
        a2c_indexed[str(article_idx)] = charge_indices

    c2a_indexed = {}
    for charge, articles in charge_to_article.items():
        charge_idx = accusation_to_idx[charge]
        article_indices = [article_to_idx[a] for a in articles]
        c2a_indexed[str(charge_idx)] = article_indices

    a2c_file = f'{output_dir}/article_to_charge_indexed.json'
    with open(a2c_file, 'w', encoding='utf-8') as f:
        json.dump(a2c_indexed, f, ensure_ascii=False, indent=2)

    c2a_file = f'{output_dir}/charge_to_article_indexed.json'
    with open(c2a_file, 'w', encoding='utf-8') as f:
        json.dump(c2a_indexed, f, ensure_ascii=False, indent=2)

    print(f"索引映射已保存:")
    print(f"   {a2c_file}")
    print(f"   {c2a_file}")

    print("\n" + "="*60)
    print("标签映射构建完成!")
    print("="*60)

    return mappings


if __name__ == "__main__":
    mappings = build_label_mappings('./dataset/train.json')

    print("\n配置更新提示:")
    print(f"   请在 config.py 中更新:")
    print(f"   num_articles = {mappings['num_articles']}")
    print(f"   num_charges = {mappings['num_accusations']}")
    print(f"   a2c_dict_file = './dataset/article_to_charge_indexed.json'")
    print(f"   c2a_dict_file = './dataset/charge_to_article_indexed.json'")
