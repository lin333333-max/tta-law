"""
评估"纯多标签样本"的性能，与K-LJP论文直接对比
"""
import os
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Subset
import json

from config import Config
from model import LegalJudgmentModel
from data_loader import get_dataloader, MappingDictionary, LegalDataset
from evaluator import Evaluator


def filter_multilabel_samples(dataset):
    """筛选出多标签样本"""
    multilabel_indices = []

    for idx in range(len(dataset)):
        sample = dataset.data[idx]
        if 'meta' in sample:
            n_articles = len(sample['meta'].get('relevant_articles', []))
            n_charges = len(sample['meta'].get('accusation', []))

            # 多标签定义：法条>1 或 罪名>1
            if n_articles > 1 or n_charges > 1:
                multilabel_indices.append(idx)

    return multilabel_indices


def main():
    config = Config()

    print("=" * 60)
    print("评估纯多标签样本（对比K-LJP论文）")
    print("=" * 60)

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    # 加载完整测试集
    test_dataset = LegalDataset(
        os.path.join(config.data_dir, config.test_file),
        tokenizer,
        config.max_seq_length,
        config.num_articles,
        config.num_charges,
        config.label_mappings_file
    )

    # 筛选多标签样本
    print("\n筛选多标签样本...")
    multilabel_indices = filter_multilabel_samples(test_dataset)
    print(f"完整测试集: {len(test_dataset)} 样本")
    print(f"多标签样本: {len(multilabel_indices)} 样本 ({len(multilabel_indices)/len(test_dataset)*100:.1f}%)")

    # 创建多标签子集
    multilabel_dataset = Subset(test_dataset, multilabel_indices)
    multilabel_loader = DataLoader(
        multilabel_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 加载模型
    model_path = os.path.join(config.save_dir, 'best_model.pt')
    model = LegalJudgmentModel(config).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ 加载模型: {model_path}")

    # 评估
    print("\n评估中...")
    evaluator = Evaluator(threshold=config.threshold)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in multilabel_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)

            output = model(input_ids, attention_mask)

            predictions = {
                'articles': output['article_probs'].cpu(),
                'charges': output['charge_probs'].cpu()
            }

            labels = {
                'articles': batch['articles'],
                'charges': batch['charges']
            }

            all_predictions.append(predictions)
            all_labels.append(labels)

    # 计算指标
    metrics = evaluator.compute_metrics(all_predictions, all_labels)

    # 加载映射字典计算对齐率
    mapping_dict = MappingDictionary(
        config.a2c_dict_file,
        config.c2a_dict_file
    )
    align_ratio = evaluator.compute_alignment_ratio(all_predictions, all_labels, mapping_dict)
    metrics['alignment_ratio'] = align_ratio

    # 打印结果
    evaluator.print_metrics(metrics, "Multi-Label Sample Only")

    # 对比K-LJP论文
    print("\n" + "=" * 60)
    print("与K-LJP论文对比")
    print("=" * 60)
    print(f"\n{'指标':<20} {'K-LJP论文':<15} {'你的模型':<15} {'差距':<15}")
    print("-" * 60)

    kljp_article_mif = 87.37
    kljp_article_maf = 61.41
    kljp_charge_mif = 88.04
    kljp_charge_maf = 58.68

    print(f"{'Article Mi-F1':<20} {kljp_article_mif:<15.2f} {metrics['article_mi_f1']*100:<15.2f} {metrics['article_mi_f1']*100-kljp_article_mif:+.2f}")
    print(f"{'Article Ma-F1':<20} {kljp_article_maf:<15.2f} {metrics['article_ma_f1']*100:<15.2f} {metrics['article_ma_f1']*100-kljp_article_maf:+.2f}")
    print(f"{'Charge Mi-F1':<20} {kljp_charge_mif:<15.2f} {metrics['charge_mi_f1']*100:<15.2f} {metrics['charge_mi_f1']*100-kljp_charge_mif:+.2f}")
    print(f"{'Charge Ma-F1':<20} {kljp_charge_maf:<15.2f} {metrics['charge_ma_f1']*100:<15.2f} {metrics['charge_ma_f1']*100-kljp_charge_maf:+.2f}")

    avg_maf = (metrics['article_ma_f1'] + metrics['charge_ma_f1']) / 2 * 100
    kljp_avg_maf = (kljp_article_maf + kljp_charge_maf) / 2
    print(f"{'Avg Ma-F1':<20} {kljp_avg_maf:<15.2f} {avg_maf:<15.2f} {avg_maf-kljp_avg_maf:+.2f}")

    # 保存结果
    results = {
        'dataset_info': {
            'total_test_samples': len(test_dataset),
            'multilabel_samples': len(multilabel_indices),
            'multilabel_ratio': len(multilabel_indices) / len(test_dataset)
        },
        'metrics': {k: float(v) for k, v in metrics.items()},
        'comparison_with_kljp': {
            'article_mif_diff': float(metrics['article_mi_f1']*100 - kljp_article_mif),
            'article_maf_diff': float(metrics['article_ma_f1']*100 - kljp_article_maf),
            'charge_mif_diff': float(metrics['charge_mi_f1']*100 - kljp_charge_mif),
            'charge_maf_diff': float(metrics['charge_ma_f1']*100 - kljp_charge_maf),
            'avg_maf_diff': float(avg_maf - kljp_avg_maf)
        }
    }

    results_path = os.path.join(config.save_dir, 'multilabel_only_evaluation.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存: {results_path}")


if __name__ == "__main__":
    main()
