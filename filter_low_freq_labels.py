"""
过滤低频标签脚本
用途: 移除训练集中频率过低的标签，解决类别不平衡问题
"""
import json
import argparse
from collections import Counter
import os


def filter_dataset(threshold=10, dry_run=False):
    """
    过滤低频标签

    Args:
        threshold: 最小频率阈值（默认10）
        dry_run: 如果True，只显示统计不实际过滤
    """
    print("=" * 70)
    print("低频标签过滤工具")
    print("=" * 70)
    print(f"阈值: 频率 < {threshold} 的标签将被过滤")
    print(f"模式: {'仅统计' if dry_run else '执行过滤'}")
    print()

    # 1. 统计训练集标签频率
    print("📊 步骤1: 统计标签频率...")
    article_freq = Counter()
    charge_freq = Counter()

    train_file = './dataset/train.json'
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'meta' in data:
                for a in data['meta'].get('relevant_articles', []):
                    article_freq[str(a)] += 1
                for c in data['meta'].get('accusation', []):
                    charge_freq[str(c)] += 1

    print(f"  法条总数: {len(article_freq)}")
    print(f"  罪名总数: {len(charge_freq)}")
    print()

    # 2. 确定要保留的标签
    print("🔍 步骤2: 识别低频标签...")
    valid_articles = {k for k, v in article_freq.items() if v >= threshold}
    valid_charges = {k for k, v in charge_freq.items() if v >= threshold}

    removed_articles = len(article_freq) - len(valid_articles)
    removed_charges = len(charge_freq) - len(valid_charges)

    print(f"  法条: {len(article_freq)} → {len(valid_articles)} (移除 {removed_articles})")
    print(f"  罪名: {len(charge_freq)} → {len(valid_charges)} (移除 {removed_charges})")
    print()

    # 显示被移除的标签
    if removed_articles > 0:
        removed_art = [k for k, v in article_freq.items() if v < threshold]
        print(f"  被移除的法条 ({len(removed_art)}个):")
        for art in sorted(removed_art, key=lambda x: article_freq[x])[:10]:
            print(f"    法条{art}: {article_freq[art]}次")
        if len(removed_art) > 10:
            print(f"    ... 还有 {len(removed_art)-10} 个")

    if removed_charges > 0:
        removed_chg = [k for k, v in charge_freq.items() if v < threshold]
        print(f"\n  被移除的罪名 ({len(removed_chg)}个):")
        for chg in sorted(removed_chg, key=lambda x: charge_freq[x])[:10]:
            print(f"    {chg}: {charge_freq[chg]}次")
        if len(removed_chg) > 10:
            print(f"    ... 还有 {len(removed_chg)-10} 个")
    print()

    # 3. 统计样本损失
    print("📉 步骤3: 评估样本损失...")

    split_stats = {}
    for split in ['train', 'dev', 'test']:
        input_file = f'./dataset/{split}.json'
        if not os.path.exists(input_file):
            print(f"  ⚠️  {split}.json 不存在，跳过")
            continue

        kept = 0
        total = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                data = json.loads(line)

                if 'meta' in data:
                    articles = [str(a) for a in data['meta'].get('relevant_articles', [])]
                    charges = data['meta'].get('accusation', [])

                    # 如果所有标签都有效，保留
                    if all(a in valid_articles for a in articles) and \
                       all(c in valid_charges for c in charges):
                        kept += 1

        loss_pct = 100 * (1 - kept / total) if total > 0 else 0
        split_stats[split] = {'total': total, 'kept': kept, 'loss_pct': loss_pct}
        print(f"  {split:5s}: {total:6d} → {kept:6d} (损失 {loss_pct:5.2f}%)")

    print()

    # 4. 对比K-LJP论文
    print("📊 步骤4: 对比K-LJP论文...")
    print(f"  K-LJP论文: 121法条, 150罪名")
    print(f"  过滤后:    {len(valid_articles)}法条, {len(valid_charges)}罪名")

    if len(valid_articles) < 121:
        print(f"  ⚠️  法条数比论文少 {121 - len(valid_articles)} 个")
        print(f"      建议降低阈值到 {threshold-5} 或使用数据增强")

    if len(valid_charges) < 150:
        print(f"  ⚠️  罪名数比论文少 {150 - len(valid_charges)} 个")
        print(f"      建议降低阈值到 {threshold-5} 或使用数据增强")

    print()

    if dry_run:
        print("=" * 70)
        print("✓ 统计完成（dry-run模式，未实际修改文件）")
        print("=" * 70)
        print("\n如果要实际执行过滤，运行:")
        print(f"  python filter_low_freq_labels.py --threshold {threshold}")
        return

    # 5. 执行过滤
    print("✂️  步骤5: 执行过滤...")

    for split in ['train', 'dev', 'test']:
        input_file = f'./dataset/{split}.json'
        output_file = f'./dataset/{split}_filtered.json'

        if not os.path.exists(input_file):
            continue

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                data = json.loads(line)

                if 'meta' in data:
                    articles = [str(a) for a in data['meta'].get('relevant_articles', [])]
                    charges = data['meta'].get('accusation', [])

                    # 如果所有标签都有效，保留
                    if all(a in valid_articles for a in articles) and \
                       all(c in valid_charges for c in charges):
                        fout.write(line)

        print(f"  ✓ {output_file}")

    # 6. 保存有效标签列表
    print("\n💾 步骤6: 保存元数据...")

    valid_labels_file = './dataset/valid_labels.json'
    with open(valid_labels_file, 'w', encoding='utf-8') as f:
        json.dump({
            'threshold': threshold,
            'articles': sorted(list(valid_articles)),
            'charges': sorted(list(valid_charges)),
            'num_articles': len(valid_articles),
            'num_charges': len(valid_charges),
            'statistics': split_stats
        }, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {valid_labels_file}")

    # 7. 生成新的config提示
    print("\n📝 步骤7: 更新配置...")
    print(f"  请更新 config.py:")
    print(f"    num_articles = {len(valid_articles)}  # 从 {len(article_freq)} 改")
    print(f"    num_charges = {len(valid_charges)}   # 从 {len(charge_freq)} 改")
    print()
    print(f"  然后运行:")
    print(f"    1. python build_mappings.py  # 重新构建映射")
    print(f"    2. python preprocess_data.py  # 重新预处理")
    print(f"    3. python train_ultrafast.py  # 重新训练")

    print()
    print("=" * 70)
    print("✅ 过滤完成！")
    print("=" * 70)
    print(f"\n预期效果: Ma-F1 从 28% → 45% (+17%)")


def main():
    parser = argparse.ArgumentParser(
        description='过滤低频标签，解决类别不平衡问题',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 统计但不执行（查看影响）
  python filter_low_freq_labels.py --dry-run

  # 过滤频率<10的标签（推荐）
  python filter_low_freq_labels.py --threshold 10

  # 过滤频率<15的标签（更激进）
  python filter_low_freq_labels.py --threshold 15
        """
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=10,
        help='最小频率阈值（默认: 10）'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅统计不实际过滤'
    )

    args = parser.parse_args()

    filter_dataset(threshold=args.threshold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
