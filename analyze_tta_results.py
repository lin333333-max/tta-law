"""
TTA三种模式结果分析报告
生成时间: 2026-07-16
"""

import json


def analyze_tta_results():
    print("=" * 80)
    print("TTA三种模式结果分析报告")
    print("=" * 80)

    # 加载数据
    modes = ['standard', 'online', 'adaptive']
    results = {}

    for mode in modes:
        with open(f'./checkpoints/tta_results/tta_results_{mode}.json', 'r') as f:
            results[mode] = json.load(f)

    # 提取指标
    print("\n" + "=" * 80)
    print("一、核心指标对比")
    print("=" * 80)

    baseline = results['standard']['baseline_metrics']

    data = []
    metrics_to_compare = [
        ('article_ma_f1', 'Article Ma-F1'),
        ('charge_ma_f1', 'Charge Ma-F1'),
        ('avg_ma_f1', 'Avg Ma-F1'),
        ('article_mi_f1', 'Article Mi-F1'),
        ('charge_mi_f1', 'Charge Mi-F1'),
        ('avg_mi_f1', 'Avg Mi-F1'),
        ('alignment_ratio', 'Alignment Ratio')
    ]

    for metric_key, metric_name in metrics_to_compare:
        row = {
            'Metric': metric_name,
            'Baseline': f"{baseline[metric_key]*100:.2f}%"
        }

        for mode in modes:
            tta_val = results[mode]['tta_metrics'][metric_key]
            baseline_val = results[mode]['baseline_metrics'][metric_key]
            diff = tta_val - baseline_val
            row[mode.capitalize()] = f"{tta_val*100:.2f}% ({diff*100:+.2f})"

        data.append(row)

    # 打印表格
    print(f"\n{'Metric':<20} {'Baseline':<12} {'Standard':<20} {'Online':<20} {'Adaptive':<20}")
    print("-" * 100)

    for row in data:
        print(f"{row['Metric']:<20} {row['Baseline']:<12} {row['Standard']:<20} {row['Online']:<20} {row['Adaptive']:<20}")

    # TTA统计
    print("\n" + "=" * 80)
    print("二、TTA更新统计")
    print("=" * 80)

    print(f"\n{'Mode':<15} {'Total':<12} {'Updated':<12} {'Update Rate':<15} {'Skipped (High Conf)':<20}")
    print("-" * 80)

    for mode in modes:
        stats = results[mode]['tta_stats']
        total = stats['total_samples']
        updated = stats['updated_samples']
        skipped = stats['skipped_high_conf']

        if total > 0:
            rate = updated / total * 100
        else:
            rate = 0

        print(f"{mode.capitalize():<15} {total:<12} {updated:<12} {rate:<15.2f}% {skipped:<20}")

    # 关键发现
    print("\n" + "=" * 80)
    print("三、关键发现")
    print("=" * 80)

    print("\n🔴 异常现象:")
    print(f"  Baseline Avg Ma-F1: {baseline['avg_ma_f1']*100:.2f}% (异常低！)")
    print(f"  之前完整测试集评估: 30.86%")
    print(f"  之前多标签样本评估: 28.48%")
    print(f"  差距: {30.86 - baseline['avg_ma_f1']*100:.2f}%")
    print("\n  可能原因:")
    print("    1. Baseline评估使用了batch_size=64 (eval_batch_size)")
    print("    2. TTA使用了batch_size=8 (tta_batch_size)")
    print("    3. 不同的batch size可能影响BatchNorm等层的行为")
    print("    4. 或者使用了不同的数据集子集")

    print("\n✅ TTA效果 (相对于异常Baseline):")
    adaptive_maf1 = results['adaptive']['tta_metrics']['avg_ma_f1']
    baseline_maf1 = baseline['avg_ma_f1']
    print(f"  Adaptive模式: {baseline_maf1*100:.2f}% → {adaptive_maf1*100:.2f}% (+{(adaptive_maf1-baseline_maf1)*100:.2f}%)")
    print(f"  提升: {(adaptive_maf1/baseline_maf1 - 1)*100:.1f}%")

    print("\n❌ TTA副作用:")
    adaptive_mif1 = results['adaptive']['tta_metrics']['avg_mi_f1']
    baseline_mif1 = baseline['avg_mi_f1']
    print(f"  Avg Mi-F1: {baseline_mif1*100:.2f}% → {adaptive_mif1*100:.2f}% ({(adaptive_mif1-baseline_mif1)*100:.2f}%)")
    print(f"  下降: {(1 - adaptive_mif1/baseline_mif1)*100:.1f}%")

    # 模式对比
    print("\n" + "=" * 80)
    print("四、三种模式对比")
    print("=" * 80)

    print("\n📌 Standard模式 (所有创新点关闭):")
    print(f"  配置: {results['standard']['config']}")
    print(f"  更新样本: {results['standard']['tta_stats']['updated_samples']} (0%)")
    print(f"  Avg Ma-F1: {results['standard']['tta_metrics']['avg_ma_f1']*100:.2f}%")
    print("  结论: 基本等同于Baseline (无TTA更新)")

    print("\n📌 Online模式 (状态累积):")
    print(f"  更新样本: {results['online']['tta_stats']['updated_samples']} (2.0%)")
    print(f"  Avg Ma-F1: {results['online']['tta_metrics']['avg_ma_f1']*100:.2f}%")
    print("  结论: 更新率太低，性能崩溃")

    print("\n📌 Adaptive模式 (动态步数) ✓ 推荐:")
    print(f"  更新样本: {results['adaptive']['tta_stats']['updated_samples']} (18.4%)")
    print(f"  Avg Ma-F1: {results['adaptive']['tta_metrics']['avg_ma_f1']*100:.2f}%")
    print("  结论: 更新率适中，性能最佳")

    # 建议
    print("\n" + "=" * 80)
    print("五、问题诊断与建议")
    print("=" * 80)

    print("\n⚠️  核心问题:")
    print("  1. Baseline的Ma-F1只有7.74%，与之前的30.86%差距巨大")
    print("  2. 这导致TTA的提升看起来很大(+23%)，但实际上是修正了异常的Baseline")
    print("  3. 需要重新用相同配置评估真实的Baseline")

    print("\n✅ 建议行动:")
    print("  1. 重新运行Baseline评估（使用tta_batch_size=8，与TTA一致）")
    print("  2. 对比真实Baseline vs TTA的效果")
    print("  3. 如果真实Baseline已经是30%，TTA可能只提升1-3%")
    print("  4. 重点应该放在优化阶段2（基础训练），而不是TTA")

    print("\n💡 TTA真实价值评估:")
    print("  假设真实Baseline = 30%:")
    print("    - TTA最佳情况: 30% → 31-33%")
    print("    - 提升幅度: 3-10%")
    print("    - 仍然远低于K-LJP论文的60%")
    print("  结论: TTA无法弥补基础模型的30%差距")

    print("\n" + "=" * 80)
    print("报告结束")
    print("=" * 80)


if __name__ == "__main__":
    analyze_tta_results()
