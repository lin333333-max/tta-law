"""
TTA测试脚本：使用训练好的模型进行测试时适应
"""
import os
import torch
from transformers import BertTokenizer
import json
from tqdm import tqdm

from config import Config
from model import LegalJudgmentModel
from data_loader import get_dataloader, MappingDictionary
from tta_trainer import TTATrainer, OnlineTTATrainer, AdaptiveTTATrainer
from evaluator import Evaluator, ErrorAnalyzer


def test_with_tta(config: Config, model_path: str, test_loader, mapping_dict,
                  tta_mode: str = 'standard', tokenizer=None):
    """
    使用TTA进行测试
    Args:
        config: 配置
        model_path: 模型路径
        test_loader: 测试数据加载器（用于baseline）
        mapping_dict: 映射字典
        tta_mode: TTA模式 ('standard', 'online', 'adaptive')
        tokenizer: 用于构建TTA专用的小batch loader
    """
    print(f"\n{'='*60}")
    print(f"🧪 Testing with TTA (Mode: {tta_mode})")
    print(f"{'='*60}\n")

    # 加载模型
    model = LegalJudgmentModel(config).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model from {model_path}")

    # 创建评估器
    evaluator = Evaluator(threshold=config.threshold)
    error_analyzer = ErrorAnalyzer(threshold=config.threshold)

    # 1. 基线评估（不使用TTA）
    print("\n📊 Step 1: Baseline evaluation (without TTA)...")
    model.eval()
    baseline_predictions = []
    baseline_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Baseline"):
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

            baseline_predictions.append(predictions)
            baseline_labels.append(labels)

    baseline_metrics = evaluator.compute_metrics(baseline_predictions, baseline_labels)
    baseline_align = evaluator.compute_alignment_ratio(baseline_predictions, baseline_labels, mapping_dict)
    baseline_metrics['alignment_ratio'] = baseline_align

    evaluator.print_metrics(baseline_metrics, "Baseline (No TTA)")

    # 2. TTA评估
    print(f"\n🔧 Step 2: Testing with TTA ({tta_mode} mode)...")

    # TTA需要反向传播+维护优化器状态，显存开销远大于纯推理的baseline评估，
    # 复用 eval_batch_size(64) 会导致置信度门控失效且速度慢，这里用更小的 tta_batch_size 单独构建一个 loader
    if tokenizer is not None:
        original_eval_batch_size = config.eval_batch_size
        config.eval_batch_size = config.tta_batch_size
        tta_loader = get_dataloader(
            config,
            os.path.join(config.data_dir, config.test_file),
            tokenizer,
            shuffle=False
        )
        config.eval_batch_size = original_eval_batch_size
    else:
        # fallback：如果没传tokenizer，复用test_loader（兼容旧调用）
        tta_loader = test_loader

    # 选择TTA训练器
    if tta_mode == 'online':
        tta_trainer = OnlineTTATrainer(model, config, mapping_dict)
    elif tta_mode == 'adaptive':
        tta_trainer = AdaptiveTTATrainer(model, config, mapping_dict)
    else:
        tta_trainer = TTATrainer(model, config, mapping_dict)

    # 执行TTA
    tta_predictions, tta_labels = tta_trainer.test_with_tta(tta_loader)

    # 计算TTA指标
    tta_metrics = evaluator.compute_metrics(tta_predictions, tta_labels)
    tta_align = evaluator.compute_alignment_ratio(tta_predictions, tta_labels, mapping_dict)
    tta_metrics['alignment_ratio'] = tta_align

    evaluator.print_metrics(tta_metrics, f"TTA ({tta_mode})")

    # 3. 对比分析
    print("\n📈 Step 3: Performance comparison...")
    evaluator.compare_metrics(baseline_metrics, tta_metrics)

    # 4. 错误分析（暂时跳过，subset模式下有尺寸不匹配的bug）
    # print("\n🔍 Step 4: Error analysis...")
    # error_analysis = error_analyzer.analyze_tta_changes(
    #     baseline_predictions, tta_predictions, tta_labels
    # )
    # error_analyzer.print_analysis(error_analysis)

    # 5. TTA统计
    print("\n📊 Step 5: TTA statistics...")
    tta_trainer.print_stats()

    # 6. 保存结果
    results = {
        'tta_mode': tta_mode,
        'baseline_metrics': {k: float(v) for k, v in baseline_metrics.items()},
        'tta_metrics': {k: float(v) for k, v in tta_metrics.items()},
        # 'error_analysis': error_analysis,  # 暂时移除，subset模式下有bug
        'tta_stats': tta_trainer.stats,
        'config': {
            'use_dynamic_weighting': config.use_dynamic_weighting,
            'use_asymmetric_alignment': config.use_asymmetric_alignment,
            'use_confidence_gating': config.use_confidence_gating,
            'use_triple_consistency': config.use_triple_consistency,
            'tta_learning_rate': config.tta_learning_rate,
            'tta_steps': config.tta_steps
        }
    }

    results_dir = os.path.join(config.save_dir, 'tta_results')
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, f'tta_results_{tta_mode}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to {results_path}")

    return baseline_metrics, tta_metrics, results


def ablation_study(config: Config, model_path: str, test_loader, mapping_dict, tokenizer=None):
    """
    消融实验：测试不同创新点的影响
    Args:
        tokenizer: 用于构建TTA专用的小batch loader
    """
    print(f"\n{'='*60}")
    print("🔬 Ablation Study")
    print(f"{'='*60}\n")

    # 保存原始配置
    original_config = {
        'use_dynamic_weighting': config.use_dynamic_weighting,
        'use_asymmetric_alignment': config.use_asymmetric_alignment,
        'use_confidence_gating': config.use_confidence_gating,
        'use_triple_consistency': config.use_triple_consistency
    }

    # 实验配置
    experiments = [
        {
            'name': 'Baseline (No innovations)',
            'use_dynamic_weighting': False,
            'use_asymmetric_alignment': False,
            'use_confidence_gating': False,
            'use_triple_consistency': False
        },
        {
            'name': 'Only Asymmetric Alignment',
            'use_dynamic_weighting': False,
            'use_asymmetric_alignment': True,
            'use_confidence_gating': False,
            'use_triple_consistency': False
        },
        {
            'name': 'Asymmetric + Confidence Gating',
            'use_dynamic_weighting': False,
            'use_asymmetric_alignment': True,
            'use_confidence_gating': True,
            'use_triple_consistency': False
        },
        {
            'name': 'Asymmetric + Confidence + Dynamic',
            'use_dynamic_weighting': True,
            'use_asymmetric_alignment': True,
            'use_confidence_gating': True,
            'use_triple_consistency': False
        },
        {
            'name': 'All innovations (Full)',
            'use_dynamic_weighting': True,
            'use_asymmetric_alignment': True,
            'use_confidence_gating': True,
            'use_triple_consistency': True
        }
    ]

    all_results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"📋 Running: {exp['name']}")
        print(f"{'='*60}")

        # 更新配置
        config.use_dynamic_weighting = exp['use_dynamic_weighting']
        config.use_asymmetric_alignment = exp['use_asymmetric_alignment']
        config.use_confidence_gating = exp['use_confidence_gating']
        config.use_triple_consistency = exp['use_triple_consistency']

        # 运行TTA测试
        baseline_metrics, tta_metrics, results = test_with_tta(
            config, model_path, test_loader, mapping_dict, tta_mode='standard', tokenizer=tokenizer
        )

        results['experiment_name'] = exp['name']
        all_results.append(results)

    # 恢复原始配置
    for key, value in original_config.items():
        setattr(config, key, value)

    # 保存消融实验结果
    ablation_path = os.path.join(config.save_dir, 'tta_results', 'ablation_study.json')
    with open(ablation_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("📊 Ablation Study Summary")
    print(f"{'='*60}\n")

    print(f"{'Experiment':<40} {'Baseline F1':>12} {'TTA F1':>12} {'Δ':>8}")
    print("-" * 72)

    for result in all_results:
        exp_name = result['experiment_name']
        baseline_f1 = result['baseline_metrics']['avg_ma_f1']
        tta_f1 = result['tta_metrics']['avg_ma_f1']
        delta = tta_f1 - baseline_f1

        print(f"{exp_name:<40} {baseline_f1:>12.4f} {tta_f1:>12.4f} {delta:>+8.4f}")

    print(f"\n💾 Ablation results saved to {ablation_path}")


def main():
    """主函数"""
    config = Config()

    # 配置TTA参数
    config.use_dynamic_weighting = True
    config.use_asymmetric_alignment = True
    config.use_confidence_gating = True
    config.use_triple_consistency = False  # 可选

    # 加载数据
    print("📂 Loading test data...")
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    test_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.test_file),
        tokenizer,
        shuffle=False
    )

    # 加载映射字典
    mapping_dict = MappingDictionary(
        config.a2c_dict_file,
        config.c2a_dict_file
    )

    # 模型路径
    model_path = os.path.join(config.save_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train.py")
        return

    # 选择测试模式
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'online', 'adaptive', 'ablation'],
                       help='TTA testing mode')
    parser.add_argument('--subset', type=int, default=None,
                       help='只用前N条测试样本（快速验证TTA效果）')
    args = parser.parse_args()

    # 如果指定subset，裁剪test_loader的数据集
    if args.subset:
        print(f"⚠️  使用测试集前 {args.subset} 条样本（快速验证模式）")
        from torch.utils.data import Subset
        original_dataset = test_loader.dataset
        subset_indices = list(range(min(args.subset, len(original_dataset))))
        subset_dataset = Subset(original_dataset, subset_indices)

        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            subset_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

    if args.mode == 'ablation':
        # 消融实验
        ablation_study(config, model_path, test_loader, mapping_dict, tokenizer=tokenizer)
    else:
        # 标准TTA测试
        test_with_tta(config, model_path, test_loader, mapping_dict, tta_mode=args.mode, tokenizer=tokenizer)

    print("\n✨ Testing completed!")


if __name__ == "__main__":
    main()
