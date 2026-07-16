"""
分析标签分布，找出类别不平衡问题
"""
import json
from collections import Counter
import numpy as np


def analyze_distribution():
    print("=" * 60)
    print("标签分布分析")
    print("=" * 60)

    # 统计训练集标签分布
    article_counts = Counter()
    charge_counts = Counter()

    with open('./dataset/train.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'meta' in data:
                for article in data['meta'].get('relevant_articles', []):
                    article_counts[str(article)] += 1
                for charge in data['meta'].get('accusation', []):
                    charge_counts[str(charge)] += 1

    print(f"\n训练集统计:")
    print(f"  样本数: 49,295")
    print(f"  法条类别数: {len(article_counts)}")
    print(f"  罪名类别数: {len(charge_counts)}")

    # 法条分布
    article_freq = sorted(article_counts.values(), reverse=True)
    print(f"\n📄 法条分布:")
    print(f"  Top-1 法条出现次数: {article_freq[0]:,}")
    print(f"  Top-10 平均: {np.mean(article_freq[:10]):,.0f}")
    print(f"  Median: {np.median(article_freq):,.0f}")
    print(f"  Bottom-10 平均: {np.mean(article_freq[-10:]):,.0f}")
    print(f"  最少: {article_freq[-1]}")
    print(f"  不平衡比: {article_freq[0] / article_freq[-1]:.1f}x")

    # 找出罕见法条
    rare_articles = [k for k, v in article_counts.items() if v < 10]
    print(f"  出现<10次的法条: {len(rare_articles)} 个")

    # 罪名分布
    charge_freq = sorted(charge_counts.values(), reverse=True)
    print(f"\n⚖️  罪名分布:")
    print(f"  Top-1 罪名出现次数: {charge_freq[0]:,}")
    print(f"  Top-10 平均: {np.mean(charge_freq[:10]):,.0f}")
    print(f"  Median: {np.median(charge_freq):,.0f}")
    print(f"  Bottom-10 平均: {np.mean(charge_freq[-10:]):,.0f}")
    print(f"  最少: {charge_freq[-1]}")
    print(f"  不平衡比: {charge_freq[0] / charge_freq[-1]:.1f}x")

    # 找出罕见罪名
    rare_charges = [k for k, v in charge_counts.items() if v < 10]
    print(f"  出现<10次的罪名: {len(rare_charges)} 个")

    # Top-10最常见
    print(f"\n🔥 Top-10 最常见法条:")
    for i, (article, count) in enumerate(article_counts.most_common(10), 1):
        print(f"  {i:2d}. 法条{article:>3s}: {count:>6,} 次")

    print(f"\n🔥 Top-10 最常见罪名:")
    for i, (charge, count) in enumerate(charge_counts.most_common(10), 1):
        print(f"  {i:2d}. {charge:<10s}: {count:>6,} 次")

    # Bottom-10最罕见
    print(f"\n❄️  Bottom-10 最罕见法条:")
    for i, (article, count) in enumerate(sorted(article_counts.items(), key=lambda x: x[1])[:10], 1):
        print(f"  {i:2d}. 法条{article:>3s}: {count:>3} 次")

    print(f"\n❄️  Bottom-10 最罕见罪名:")
    for i, (charge, count) in enumerate(sorted(charge_counts.items(), key=lambda x: x[1])[:10], 1):
        print(f"  {i:2d}. {charge:<10s}: {count:>3} 次")

    # 保存详细统计
    stats = {
        'articles': {
            'total_classes': len(article_counts),
            'max_freq': int(article_freq[0]),
            'min_freq': int(article_freq[-1]),
            'median_freq': float(np.median(article_freq)),
            'imbalance_ratio': float(article_freq[0] / article_freq[-1]),
            'rare_classes_count': len(rare_articles),
            'distribution': dict(article_counts)
        },
        'charges': {
            'total_classes': len(charge_counts),
            'max_freq': int(charge_freq[0]),
            'min_freq': int(charge_freq[-1]),
            'median_freq': float(np.median(charge_freq)),
            'imbalance_ratio': float(charge_freq[0] / charge_freq[-1]),
            'rare_classes_count': len(rare_charges),
            'distribution': dict(charge_counts)
        }
    }

    with open('./checkpoints/label_distribution_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 详细统计已保存: ./checkpoints/label_distribution_analysis.json")


if __name__ == "__main__":
    analyze_distribution()
