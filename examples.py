"""
使用示例：演示如何使用TTA-LJP系统
"""

# ============================================
# 示例1：基础训练和测试
# ============================================

def example_basic_train_and_test():
    """基础训练和测试流程"""
    from config import Config
    from model import LegalJudgmentModel
    from transformers import BertTokenizer
    from data_loader import get_dataloader, MappingDictionary
    import torch
    import os

    print("="*60)
    print("示例1：基础训练和测试")
    print("="*60)

    # 1. 加载配置
    config = Config()

    # 2. 准备数据
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    train_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.train_file),
        tokenizer,
        shuffle=True
    )

    # 3. 初始化模型
    model = LegalJudgmentModel(config).to(config.device)

    # 4. 训练（简化示例）
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)

        # 前向传播
        output = model(input_ids, attention_mask)

        # 计算损失并优化
        # ... (详见train.py)

        break  # 示例只运行一个batch

    print("✅ 训练完成")


# ============================================
# 示例2：使用TTA进行测试
# ============================================

def example_tta_testing():
    """TTA测试示例"""
    from config import Config
    from model import LegalJudgmentModel
    from tta_trainer import TTATrainer
    from data_loader import MappingDictionary
    import torch

    print("\n" + "="*60)
    print("示例2：TTA测试")
    print("="*60)

    config = Config()

    # 启用创新点
    config.use_dynamic_weighting = True
    config.use_asymmetric_alignment = True
    config.use_confidence_gating = True

    # 加载模型
    model = LegalJudgmentModel(config).to(config.device)

    # 加载映射字典
    mapping_dict = MappingDictionary(
        config.a2c_dict_file,
        config.c2a_dict_file
    )

    # 创建TTA训练器
    tta_trainer = TTATrainer(model, config, mapping_dict)

    # 模拟一个batch
    batch = {
        'input_ids': torch.randint(0, 1000, (4, 128)).to(config.device),
        'attention_mask': torch.ones(4, 128).to(config.device)
    }

    # 执行TTA更新
    output = tta_trainer.tta_update_sample(batch, num_steps=5)

    print(f"✅ TTA完成")
    print(f"   法条预测形状: {output['article_probs'].shape}")
    print(f"   罪名预测形状: {output['charge_probs'].shape}")

    # 打印统计信息
    tta_trainer.print_stats()


# ============================================
# 示例3：自定义创新点配置
# ============================================

def example_custom_innovations():
    """自定义创新点配置"""
    from config import Config

    print("\n" + "="*60)
    print("示例3：自定义创新点配置")
    print("="*60)

    # 配置1：只使用不对称更新
    config1 = Config()
    config1.use_dynamic_weighting = False
    config1.use_asymmetric_alignment = True
    config1.use_confidence_gating = False
    config1.use_triple_consistency = False
    print("\n配置1 - 只使用不对称更新:")
    print(f"  Alpha (罪名→法条): {config1.alpha}")
    print(f"  Beta (法条→罪名): {config1.beta}")

    # 配置2：不对称 + 置信度门控
    config2 = Config()
    config2.use_dynamic_weighting = False
    config2.use_asymmetric_alignment = True
    config2.use_confidence_gating = True
    config2.use_triple_consistency = False
    print("\n配置2 - 不对称 + 置信度门控:")
    print(f"  高置信度阈值: {config2.confidence_high_threshold}")
    print(f"  低置信度阈值: {config2.confidence_low_threshold}")

    # 配置3：全部创新点
    config3 = Config()
    config3.use_dynamic_weighting = True
    config3.use_asymmetric_alignment = True
    config3.use_confidence_gating = True
    config3.use_triple_consistency = True
    print("\n配置3 - 全部创新点:")
    print(f"  动态加权: ✅")
    print(f"  不对称更新: ✅")
    print(f"  置信度门控: ✅")
    print(f"  三重一致性: ✅")


# ============================================
# 示例4：评估和分析
# ============================================

def example_evaluation():
    """评估和分析示例"""
    from evaluator import Evaluator, ErrorAnalyzer
    import torch

    print("\n" + "="*60)
    print("示例4：评估和分析")
    print("="*60)

    # 创建评估器
    evaluator = Evaluator(threshold=0.5)
    error_analyzer = ErrorAnalyzer(threshold=0.5)

    # 模拟预测结果
    predictions = [{
        'articles': torch.sigmoid(torch.randn(10, 121)),
        'charges': torch.sigmoid(torch.randn(10, 150))
    }]

    labels = [{
        'articles': torch.randint(0, 2, (10, 121)).float(),
        'charges': torch.randint(0, 2, (10, 150)).float()
    }]

    # 计算指标
    metrics = evaluator.compute_metrics(predictions, labels)

    print("\n评估指标:")
    print(f"  Article Macro-F1: {metrics['article_ma_f1']:.4f}")
    print(f"  Charge Macro-F1: {metrics['charge_ma_f1']:.4f}")
    print(f"  Average Macro-F1: {metrics['avg_ma_f1']:.4f}")

    # 错误分析
    baseline_preds = predictions  # 简化示例
    tta_preds = predictions

    analysis = error_analyzer.analyze_tta_changes(baseline_preds, tta_preds, labels)
    print("\n错误分析:")
    print(f"  总样本数: {analysis['articles']['total_samples']}")
    print(f"  改变的样本: {analysis['articles']['num_changed']}")


# ============================================
# 示例5：消融实验
# ============================================

def example_ablation_study():
    """消融实验示例"""
    from config import Config

    print("\n" + "="*60)
    print("示例5：消融实验配置")
    print("="*60)

    experiments = [
        {
            'name': '基线',
            'dynamic': False,
            'asymmetric': False,
            'gating': False,
            'triple': False
        },
        {
            'name': '+不对称',
            'dynamic': False,
            'asymmetric': True,
            'gating': False,
            'triple': False
        },
        {
            'name': '+不对称+门控',
            'dynamic': False,
            'asymmetric': True,
            'gating': True,
            'triple': False
        },
        {
            'name': '+不对称+门控+动态',
            'dynamic': True,
            'asymmetric': True,
            'gating': True,
            'triple': False
        },
        {
            'name': '全部',
            'dynamic': True,
            'asymmetric': True,
            'gating': True,
            'triple': True
        }
    ]

    print("\n消融实验配置:")
    print(f"{'实验':<20} {'动态':<6} {'不对称':<8} {'门控':<6} {'三重':<6}")
    print("-" * 50)

    for exp in experiments:
        print(f"{exp['name']:<20} "
              f"{'✅' if exp['dynamic'] else '❌':<6} "
              f"{'✅' if exp['asymmetric'] else '❌':<8} "
              f"{'✅' if exp['gating'] else '❌':<6} "
              f"{'✅' if exp['triple'] else '❌':<6}")


# ============================================
# 主函数
# ============================================

def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("🎓 TTA-LJP 使用示例")
    print("="*60)

    try:
        # 示例3：配置示例（不需要数据）
        example_custom_innovations()

        # 示例4：评估示例（不需要真实数据）
        example_evaluation()

        # 示例5：消融实验配置
        example_ablation_study()

        print("\n" + "="*60)
        print("✅ 所有示例运行完成！")
        print("="*60)

        print("\n📚 更多示例请查看:")
        print("  - train.py: 完整训练流程")
        print("  - test_tta.py: TTA测试流程")
        print("  - run_experiment.py: 一键实验脚本")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
