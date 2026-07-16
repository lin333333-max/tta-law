"""
增强的评估模块：包含更详细的指标
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_score, recall_score
from typing import Dict, List


class EnhancedEvaluator:
    """增强的评估器（包含更多指标）"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_metrics(self, predictions: List[Dict], labels: List[Dict]) -> Dict:
        """
        计算全面的评估指标
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

        metrics = {}

        # ============ 法条预测指标 ============
        # F1 分数
        metrics['article_mi_f1'] = f1_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_f1'] = f1_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # Jaccard 系数
        metrics['article_mi_jaccard'] = jaccard_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_jaccard'] = jaccard_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # 精确率（Precision）
        metrics['article_mi_precision'] = precision_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_precision'] = precision_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # 召回率（Recall）
        metrics['article_mi_recall'] = recall_score(article_labels_np, article_preds_binary, average='micro', zero_division=0)
        metrics['article_ma_recall'] = recall_score(article_labels_np, article_preds_binary, average='macro', zero_division=0)

        # ============ 罪名预测指标 ============
        # F1 分数
        metrics['charge_mi_f1'] = f1_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_f1'] = f1_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)

        # Jaccard 系数
        metrics['charge_mi_jaccard'] = jaccard_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_jaccard'] = jaccard_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)

        # 精确率（Precision）
        metrics['charge_mi_precision'] = precision_score(charge_labels_np, charge_preds_binary, average='micro', zero_division=0)
        metrics['charge_ma_precision'] = precision_score(charge_labels_np, charge_preds_binary, average='macro', zero_division=0)

        # 召回率（Recall）
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

    def print_metrics(self, metrics: Dict, title: str = "Evaluation Results"):
        """打印详细的指标"""
        print("\n" + "="*60)
        print(f"{title}")
        print("="*60)

        print("\n📄 Law Article Prediction:")
        print(f"  Micro-F1: {metrics['article_mi_f1']:.4f}")
        print(f"  Macro-F1: {metrics['article_ma_f1']:.4f}")
        print(f"  Micro-Precision: {metrics['article_mi_precision']:.4f}")
        print(f"  Micro-Recall: {metrics['article_mi_recall']:.4f}")
        print(f"  Exact Match: {metrics['article_exact_match']:.4f}")
        print(f"  Hamming Acc: {metrics['article_hamming_acc']:.4f}")

        print("\n⚖  Charge Prediction:")
        print(f"  Micro-F1: {metrics['charge_mi_f1']:.4f}")
        print(f"  Macro-F1: {metrics['charge_ma_f1']:.4f}")
        print(f"  Micro-Precision: {metrics['charge_mi_precision']:.4f}")
        print(f"  Micro-Recall: {metrics['charge_mi_recall']:.4f}")
        print(f"  Exact Match: {metrics['charge_exact_match']:.4f}")
        print(f"  Hamming Acc: {metrics['charge_hamming_acc']:.4f}")

        print("\n📊 Average:")
        print(f"  Avg Micro-F1: {metrics['avg_mi_f1']:.4f}")
        print(f"  Avg Macro-F1: {metrics['avg_ma_f1']:.4f}")
        print(f"  Avg Micro-Precision: {metrics['avg_mi_precision']:.4f}")
        print(f"  Avg Micro-Recall: {metrics['avg_mi_recall']:.4f}")

        if 'alignment_ratio' in metrics:
            print(f"\n🔗 Alignment Ratio: {metrics['alignment_ratio']:.4f}")

        print("\n" + "="*60 + "\n")

    def compute_alignment_ratio(self, predictions: List[Dict], labels: List[Dict],
                               mapping_dict) -> float:
        """计算对齐率（这是对齐指标）"""
        all_article_preds = torch.cat([p['articles'] for p in predictions], dim=0)
        all_charge_preds = torch.cat([p['charges'] for p in predictions], dim=0)

        article_binary = (all_article_preds > self.threshold).float()
        charge_binary = (all_charge_preds > self.threshold).float()

        total_align_score = 0.0
        num_samples = article_binary.size(0)

        for i in range(num_samples):
            articles = article_binary[i]
            charges = charge_binary[i]

            aligned_count = 0
            total_count = 0

            for article_idx in range(len(articles)):
                if articles[article_idx] > 0:
                    total_count += 1
                    mapped_charges = mapping_dict.get_mapped_charges(article_idx)
                    for charge_idx in mapped_charges:
                        charge_idx = int(charge_idx) if isinstance(charge_idx, str) else charge_idx
                        if charge_idx < len(charges) and charges[charge_idx] > 0:
                            aligned_count += 1
                            break

            for charge_idx in range(len(charges)):
                if charges[charge_idx] > 0:
                    total_count += 1
                    mapped_articles = mapping_dict.get_mapped_articles(charge_idx)
                    for article_idx in mapped_articles:
                        article_idx = int(article_idx) if isinstance(article_idx, str) else article_idx
                        if article_idx < len(articles) and articles[article_idx] > 0:
                            aligned_count += 1
                            break

            if total_count > 0:
                total_align_score += aligned_count / total_count

        return total_align_score / num_samples if num_samples > 0 else 0.0


# 向后兼容
Evaluator = EnhancedEvaluator
