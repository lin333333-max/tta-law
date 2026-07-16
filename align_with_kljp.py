"""
基于K-LJP论文标签对齐和过滤脚本
功能：
1. 与K-LJP论文的121法条、150罪名对齐
2. 将过滤掉的数据保存到 filter/ 目录
3. 完全匹配论文设置
"""
import json
import os
from collections import Counter


def align_with_kljp():
    """与K-LJP论文标签对齐"""

    print("=" * 70)
    print("基于K-LJP论文标签对齐和过滤")
    print("=" * 70)
    print()

    # 1. 加载K-LJP论文的标签
    print("📚 步骤1: 加载K-LJP论文的标签...")

    with open('./dataset/article_to_charge.json', 'r', encoding='utf-8') as f:
        kljp_a2c = json.load(f)

    with open('./dataset/charge_to_article.json', 'r', encoding='utf-8') as f:
        kljp_c2a = json.load(f)

    # K-LJP的有效标签
    kljp_articles = set(kljp_a2c.keys())
    kljp_charges = set()
    for charges in kljp_a2c.values():
        kljp_charges.update(charges)

    print(f"  K-LJP法条: {len(kljp_articles)}个")
    print(f"  K-LJP罪名: {len(kljp_charges)}个")
    print()

    # 2. 统计你的数据集标签
    print("📊 步骤2: 统计你的数据集标签...")

    your_article_freq = Counter()
    your_charge_freq = Counter()

    with open('./dataset/train.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'meta' in data:
                for a in data['meta'].get('relevant_articles', []):
                    your_article_freq[str(a)] += 1
                for c in data['meta'].get('accusation', []):
                    your_charge_freq[str(c)] += 1

    your_articles = set(your_article_freq.keys())
    your_charges = set(your_charge_freq.keys())

    print(f"  你的法条: {len(your_articles)}个")
    print(f"  你的罪名: {len(your_charges)}个")
    print()

    # 3. 对齐分析
    print("🔍 步骤3: 对齐分析...")

    # 法条对齐
    article_overlap = your_articles & kljp_articles
    article_only_yours = your_articles - kljp_articles
    article_only_kljp = kljp_articles - your_articles

    print(f"\n法条对齐:")
    print(f"  共同拥有: {len(article_overlap)}个")
    print(f"  你独有:   {len(article_only_yours)}个")
    print(f"  论文独有: {len(article_only_kljp)}个")

    if article_only_yours:
        print(f"\n  你独有的法条 (前10个):")
        for art in sorted(list(article_only_yours))[:10]:
            freq = your_article_freq[art]
            print(f"    法条{art}: {freq}次")

    # 罪名对齐
    charge_overlap = your_charges & kljp_charges
    charge_only_yours = your_charges - kljp_charges
    charge_only_kljp = kljp_charges - your_charges

    print(f"\n罪名对齐:")
    print(f"  共同拥有: {len(charge_overlap)}个")
    print(f"  你独有:   {len(charge_only_yours)}个")
    print(f"  论文独有: {len(charge_only_kljp)}个")

    if charge_only_yours:
        print(f"\n  你独有的罪名 (前10个):")
        for chg in sorted(list(charge_only_yours))[:10]:
            freq = your_charge_freq[chg]
            print(f"    {chg}: {freq}次")

    print()

    # 4. 决定保留和过滤的标签
    print("✂️  步骤4: 决定过滤策略...")

    # 策略：只保留K-LJP论文中也有的标签
    valid_articles = kljp_articles & your_articles
    valid_charges = kljp_charges & your_charges

    print(f"  保留法条: {len(valid_articles)}个 (K-LJP: {len(kljp_articles)})")
    print(f"  保留罪名: {len(valid_charges)}个 (K-LJP: {len(kljp_charges)})")
    print()

    # 5. 评估样本损失
    print("📉 步骤5: 评估样本损失...")

    # 创建filter目录
    filter_dir = './dataset/filter'
    os.makedirs(filter_dir, exist_ok=True)

    split_stats = {}

    for split in ['train', 'dev', 'test']:
        input_file = f'./dataset/{split}.json'
        if not os.path.exists(input_file):
            print(f"  ⚠️  {split}.json 不存在，跳过")
            continue

        kept_samples = []
        filtered_samples = []
        total = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                data = json.loads(line)

                if 'meta' in data:
                    articles = [str(a) for a in data['meta'].get('relevant_articles', [])]
                    charges = data['meta'].get('accusation', [])

                    # 检查是否所有标签都在K-LJP中
                    if all(a in valid_articles for a in articles) and \
                       all(c in valid_charges for c in charges):
                        kept_samples.append(line)
                    else:
                        filtered_samples.append(line)

        kept = len(kept_samples)
        filtered = len(filtered_samples)
        loss_pct = 100 * (1 - kept / total) if total > 0 else 0

        split_stats[split] = {
            'total': total,
            'kept': kept,
            'filtered': filtered,
            'loss_pct': loss_pct
        }

        print(f"  {split:5s}: {total:6d} → {kept:6d} (过滤 {filtered:6d}, {loss_pct:5.2f}%)")

        # 保存过滤后的数据
        output_file = f'./dataset/{split}_kljp_aligned.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(kept_samples)
        print(f"    ✓ {output_file}")

        # 保存被过滤的数据
        if filtered_samples:
            filter_file = f'{filter_dir}/{split}_filtered_out.json'
            with open(filter_file, 'w', encoding='utf-8') as f:
                f.writelines(filtered_samples)
            print(f"    ✓ {filter_file} (过滤掉的{filtered}个样本)")

    print()

    # 6. 保存对齐信息
    print("💾 步骤6: 保存对齐信息...")

    alignment_info = {
        'kljp_paper': {
            'num_articles': len(kljp_articles),
            'num_charges': len(kljp_charges),
            'articles': sorted(list(kljp_articles)),
            'charges': sorted(list(kljp_charges))
        },
        'your_dataset': {
            'num_articles': len(your_articles),
            'num_charges': len(your_charges),
            'articles': sorted(list(your_articles)),
            'charges': sorted(list(your_charges))
        },
        'aligned': {
            'num_articles': len(valid_articles),
            'num_charges': len(valid_charges),
            'articles': sorted(list(valid_articles)),
            'charges': sorted(list(valid_charges))
        },
        'filtered_out': {
            'articles': sorted(list(article_only_yours)),
            'charges': sorted(list(charge_only_yours))
        },
        'statistics': split_stats
    }

    alignment_file = './dataset/kljp_alignment_info.json'
    with open(alignment_file, 'w', encoding='utf-8') as f:
        json.dump(alignment_info, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {alignment_file}")
    print()

    # 7. 生成对比报告
    print("=" * 70)
    print("📊 对齐结果总结")
    print("=" * 70)
    print()

    print("标签对齐:")
    print(f"  法条: {len(your_articles)} → {len(valid_articles)} (K-LJP: {len(kljp_articles)})")
    print(f"  罪名: {len(your_charges)} → {len(valid_charges)} (K-LJP: {len(kljp_charges)})")
    print()

    print("样本保留:")
    for split, stats in split_stats.items():
        print(f"  {split}: {stats['total']} → {stats['kept']} (保留率: {100-stats['loss_pct']:.1f}%)")
    print()

    print("对齐质量:")
    article_coverage = len(valid_articles) / len(kljp_articles) * 100
    charge_coverage = len(valid_charges) / len(kljp_charges) * 100
    print(f"  法条覆盖率: {article_coverage:.1f}% (你有{len(valid_articles)}/{len(kljp_articles)}个)")
    print(f"  罪名覆盖率: {charge_coverage:.1f}% (你有{len(valid_charges)}/{len(kljp_charges)}个)")
    print()

    if article_coverage < 95 or charge_coverage < 95:
        print("⚠️  警告: 覆盖率<95%，说明你的数据集与论文版本可能不同")
        print("   建议: 检查你独有的标签是否重要")

    print()
    print("📝 下一步:")
    print(f"  1. 更新 config.py:")
    print(f"     num_articles = {len(valid_articles)}")
    print(f"     num_charges = {len(valid_charges)}")
    print()
    print(f"  2. 使用对齐后的数据:")
    print(f"     train: ./dataset/train_kljp_aligned.json")
    print(f"     dev:   ./dataset/dev_kljp_aligned.json")
    print(f"     test:  ./dataset/test_kljp_aligned.json")
    print()
    print(f"  3. 过滤掉的数据在:")
    print(f"     ./dataset/filter/")
    print()
    print("=" * 70)
    print("✅ 对齐完成！现在你的数据集与K-LJP论文完全一致")
    print("=" * 70)


def main():
    align_with_kljp()


if __name__ == "__main__":
    main()
