"""
评估模块：计算各种指标
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_score, recall_score
from typing import Dict, List, Tuple


class Evaluator:
    """评估器"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_metrics(self, predictions: List[Dict], labels: List[Dict]) -> Dict:
        """
        计算评估指标
        Args:
            predictions: 预测结果列表
            labels: 真实标签列表
        Returns:
            指标字典
        """
        # 合并所有batch
        all_article_preds = torch.cat([p['articles'] for p in predictions], dim=0)
        all_charge_preds = torch.cat([p['charges'] for p in predictions], dim=0)
        all_article_labels = torch.cat([l['articles'] for l in labels], dim=0)
        all_charge_labels = torch.cat([l['charges'] for l in labels], dim=0)

        # 二值化预测
        article_preds_binary = (all_article_preds > self.threshold).float().numpy()
        charge_preds_binary = (all_charge_preds > self.threshold).float().numpy()
        article_labels_np = all_article_labels.numpy()
        charge_labels_np = all_charge_labels.numpy()

        # 计算指标
        metrics = {}

        # ============ 法条预测指标 ============
        # F1 分数
        metrics['article_mi_f1'] = f1_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_f1'] = f1_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # Jaccard 系数
        metrics['article_mi_jaccard'] = jaccard_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_jaccard'] = jaccard_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # 精确率和召回率
        metrics['article_mi_precision'] = precision_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_precision'] = precision_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)
        metrics['article_mi_recall'] = recall_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_recall'] = recall_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # ============ 罪名预测指标 ============
        # F1 分数
        metrics['charge_mi_f1'] = f1_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_f1'] = f1_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)

        # Jaccard 系数
        metrics['charge_mi_jaccard'] = jaccard_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_jaccard'] = jaccard_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)

        # 精确率和召回率
        metrics['charge_mi_precision'] = precision_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_precision'] = precision_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)
        metrics['charge_mi_recall'] = recall_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_recall'] = recall_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)

        # ============ 样本级准确率 ============
        # Exact Match（完全匹配率）
        article_exact_match = np.mean(np.all(article_preds_binary == article_labels_np, axis=1))
        charge_exact_match = np.mean(np.all(charge_preds_binary == charge_labels_np, axis=1))
        metrics['article_exact_match'] = article_exact_match
        metrics['charge_exact_match'] = charge_exact_match

        # Hamming Accuracy（逐标签准确率）
        article_hamming_acc = accuracy_score(article_labels_np.flatten(), article_preds_binary.flatten())
        charge_hamming_acc = accuracy_score(charge_labels_np.flatten(), charge_preds_binary.flatten())
        metrics['article_hamming_acc'] = article_hamming_acc
        metrics['charge_hamming_acc'] = charge_hamming_acc

        # ============ 平均指标 ============
        metrics['avg_mi_f1'] = (metrics['article_mi_f1'] + metrics['charge_mi_f1']) / 2
        metrics['avg_ma_f1'] = (metrics['article_ma_f1'] + metrics['charge_ma_f1']) / 2
        metrics['avg_mi_precision'] = (metrics['article_mi_precision'] + metrics['charge_mi_precision']) / 2
        metrics['avg_mi_recall'] = (metrics['article_mi_recall'] + metrics['charge_mi_recall']) / 2

        return metrics

    def compute_alignment_ratio(self, predictions: List[Dict], labels: List[Dict],
                               mapping_dict) -> float:
        """
        计算对齐率
        Args:
            predictions: 预测结果列表
            labels: 真实标签列表（实际不使用，只用于格式统一）
            mapping_dict: 映射字典
        Returns:
            对齐率
        """
        all_article_preds = torch.cat([p['articles'] for p in predictions], dim=0)
        all_charge_preds = torch.cat([p['charges'] for p in predictions], dim=0)

        # 二值化
        article_binary = (all_article_preds > self.threshold).float()
        charge_binary = (all_charge_preds > self.threshold).float()

        total_align_score = 0.0
        num_samples = article_binary.size(0)

        for i in range(num_samples):
            articles = article_binary[i]
            charges = charge_binary[i]

            # 计算对齐标签数
            aligned_count = 0
            total_count = 0

            # 检查每个预测的法条
            for article_idx in range(len(articles)):
                if articles[article_idx] > 0:
                    total_count += 1
                    # 检查其映射的罪名是否被预测
                    mapped_charges = mapping_dict.get_mapped_charges(article_idx)
                    for charge_idx in mapped_charges:
                        if charge_idx < len(charges) and charges[charge_idx] > 0:
                            aligned_count += 1
                            break

            # 检查每个预测的罪名
            for charge_idx in range(len(charges)):
                if charges[charge_idx] > 0:
                    total_count += 1
                    # 检查其映射的法条是否被预测
                    mapped_articles = mapping_dict.get_mapped_articles(charge_idx)
                    for article_idx in mapped_articles:
                        if article_idx < len(articles) and articles[article_idx] > 0:
                            aligned_count += 1
                            break

            if total_count > 0:
                align_score = aligned_count / total_count
            else:
                align_score = 0.0

            total_align_score += align_score

        avg_align_ratio = total_align_score / num_samples if num_samples > 0 else 0.0
        return avg_align_ratio

    def print_metrics(self, metrics: Dict, prefix: str = ""):
        """
        打印指标（基于 K-LJP 论文的标准指标）
        参考：Legal Judgment Prediction based on Knowledge-enhanced Multi-Task
        """
        print(f"\n{'='*60}")
        print(f"{prefix} Evaluation Results")
        print(f"{'='*60}")

        # 法条预测（论文标准指标：Mi-F, Ma-F, Mi-J, Ma-J）
        print(f"\n📄 Law Article Prediction:")
        print(f"  Micro-F1 (Mi-F): {metrics['article_mi_f1']:.4f}")
        print(f"  Macro-F1 (Ma-F): {metrics['article_ma_f1']:.4f}")
        print(f"  Micro-Jaccard (Mi-J): {metrics['article_mi_jaccard']:.4f}")
        print(f"  Macro-Jaccard (Ma-J): {metrics['article_ma_jaccard']:.4f}")

        # 罪名预测（论文标准指标：Mi-F, Ma-F, Mi-J, Ma-J）
        print(f"\n⚖️  Charge Prediction:")
        print(f"  Micro-F1 (Mi-F): {metrics['charge_mi_f1']:.4f}")
        print(f"  Macro-F1 (Ma-F): {metrics['charge_ma_f1']:.4f}")
        print(f"  Micro-Jaccard (Mi-J): {metrics['charge_mi_jaccard']:.4f}")
        print(f"  Macro-Jaccard (Ma-J): {metrics['charge_ma_jaccard']:.4f}")

        # 平均指标
        print(f"\n📊 Average:")
        print(f"  Avg Micro-F1: {metrics['avg_mi_f1']:.4f}")
        print(f"  Avg Macro-F1: {metrics['avg_ma_f1']:.4f}")

        # 对齐率（创新点指标，来自 K-LJP 论文公式 9）
        if 'alignment_ratio' in metrics:
            print(f"\n🔗 Alignment Ratio (Align): {metrics['alignment_ratio']:.4f}")
            print(f"    Formula: (|a∩C2A(c)| + |c∩A2C(a)|) / (|a| + |c|)")

        print(f"\n{'='*60}\n")

    def compare_metrics(self, baseline_metrics: Dict, tta_metrics: Dict):
        """比较基线和TTA结果"""
        print(f"\n{'='*60}")
        print("📈 Performance Comparison: Baseline vs TTA")
        print(f"{'='*60}")

        metrics_to_compare = [
            ('article_mi_f1', 'Article Micro-F1'),
            ('article_ma_f1', 'Article Macro-F1'),
            ('charge_mi_f1', 'Charge Micro-F1'),
            ('charge_ma_f1', 'Charge Macro-F1'),
            ('avg_mi_f1', 'Average Micro-F1'),
            ('avg_ma_f1', 'Average Macro-F1')
        ]

        for key, name in metrics_to_compare:
            baseline = baseline_metrics[key]
            tta = tta_metrics[key]
            diff = tta - baseline
            diff_pct = (diff / baseline * 100) if baseline > 0 else 0

            symbol = "🔺" if diff > 0 else ("🔻" if diff < 0 else "➖")
            print(f"{symbol} {name:25s}: {baseline:.4f} → {tta:.4f} ({diff:+.4f}, {diff_pct:+.2f}%)")

        print(f"{'='*60}\n")


class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def analyze_tta_changes(self, baseline_preds: List[Dict], tta_preds: List[Dict],
                           labels: List[Dict]) -> Dict:
        """
        分析TTA导致的预测变化
        Args:
            baseline_preds: 基线预测
            tta_preds: TTA预测
            labels: 真实标签
        Returns:
            分析结果
        """
        # 合并数据
        baseline_articles = torch.cat([p['articles'] for p in baseline_preds], dim=0)
        baseline_charges = torch.cat([p['charges'] for p in baseline_preds], dim=0)
        tta_articles = torch.cat([p['articles'] for p in tta_preds], dim=0)
        tta_charges = torch.cat([p['charges'] for p in tta_preds], dim=0)
        true_articles = torch.cat([l['articles'] for l in labels], dim=0)
        true_charges = torch.cat([l['charges'] for l in labels], dim=0)

        # 二值化
        baseline_articles_bin = (baseline_articles > self.threshold).float()
        baseline_charges_bin = (baseline_charges > self.threshold).float()
        tta_articles_bin = (tta_articles > self.threshold).float()
        tta_charges_bin = (tta_charges > self.threshold).float()

        results = {
            'articles': self._analyze_task_changes(
                baseline_articles_bin, tta_articles_bin, true_articles
            ),
            'charges': self._analyze_task_changes(
                baseline_charges_bin, tta_charges_bin, true_charges
            )
        }

        return results

    def _analyze_task_changes(self, baseline: torch.Tensor, tta: torch.Tensor,
                              truth: torch.Tensor) -> Dict:
        """分析单个任务的变化"""
        num_samples = baseline.size(0)

        # 检测变化
        changed = (baseline != tta).any(dim=1)
        num_changed = changed.sum().item()

        # 统计变化类型
        correct_to_correct = 0
        correct_to_wrong = 0
        wrong_to_correct = 0
        wrong_to_wrong = 0

        for i in range(num_samples):
            if changed[i]:
                baseline_correct = (baseline[i] == truth[i]).all().item()
                tta_correct = (tta[i] == truth[i]).all().item()

                if baseline_correct and tta_correct:
                    correct_to_correct += 1
                elif baseline_correct and not tta_correct:
                    correct_to_wrong += 1
                elif not baseline_correct and tta_correct:
                    wrong_to_correct += 1
                else:
                    wrong_to_wrong += 1

        return {
            'total_samples': num_samples,
            'num_changed': num_changed,
            'change_rate': num_changed / num_samples,
            'correct_to_correct': correct_to_correct,
            'correct_to_wrong': correct_to_wrong,
            'wrong_to_correct': wrong_to_correct,
            'wrong_to_wrong': wrong_to_wrong,
            'net_improvement': wrong_to_correct - correct_to_wrong
        }

    def print_analysis(self, analysis: Dict):
        """打印错误分析结果"""
        print(f"\n{'='*60}")
        print("🔍 TTA Change Analysis")
        print(f"{'='*60}")

        for task_name, stats in analysis.items():
            print(f"\n{task_name.upper()}:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Changed samples: {stats['num_changed']} ({stats['change_rate']:.2%})")
            print(f"\n  Change breakdown:")
            print(f"    ✅ Correct → Correct: {stats['correct_to_correct']}")
            print(f"    ✅➡️❌ Correct → Wrong: {stats['correct_to_wrong']}")
            print(f"    ❌➡️✅ Wrong → Correct: {stats['wrong_to_correct']}")
            print(f"    ❌ Wrong → Wrong: {stats['wrong_to_wrong']}")
            print(f"\n  📊Net improvement: {stats['net_improvement']:+d} samples")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # 测试代码
    evaluator = Evaluator()

    # 创建虚拟数据
    predictions = [
        {
            'articles': torch.sigmoid(torch.randn(4, 121)),
            'charges': torch.sigmoid(torch.randn(4, 150))
        }
    ]

    labels = [
        {
            'articles': torch.randint(0, 2, (4, 121)).float(),
            'charges': torch.randint(0, 2, (4, 150)).float()
        }
    ]

    # 测试评估
    metrics = evaluator.compute_metrics(predictions, labels)
    evaluator.print_metrics(metrics, "Test")

    print("Evaluator test passed!")
