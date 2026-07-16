"""
快速实验脚本：一键运行完整实验流程
包括训练、TTA测试、消融实验和结果分析
"""
import os
import sys
import argparse
import json
from pathlib import Path

from config import Config
from train import main as train_main
from test_tta import test_with_tta, ablation_study
from transformers import BertTokenizer
from data_loader import get_dataloader, MappingDictionary, create_mapping_dict_from_cail


def setup_experiment(config: Config, create_dicts: bool = False):
    """设置实验环境"""
    print("\n" + "="*60)
    print("🔧 Setting up experiment environment")
    print("="*60)

    # 创建必要的目录
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)

    print(f"✅ Created directories:")
    print(f"   - Data: {config.data_dir}")
    print(f"   - Logs: {config.log_dir}")
    print(f"   - Checkpoints: {config.save_dir}")

    # 构建映射字典（如果需要）
    if create_dicts:
        print("\n📚 Creating mapping dictionaries...")
        train_file = os.path.join(config.data_dir, config.train_file)

        if os.path.exists(train_file):
            create_mapping_dict_from_cail(
                train_file,
                config.a2c_dict_file,
                config.c2a_dict_file
            )
            print("✅ Mapping dictionaries created!")
        else:
            print(f"⚠️  Warning: Training file not found at {train_file}")
            print("   Please ensure your data files are in the correct location.")
    else:
        print("\n📚 Using existing mapping dictionaries")

    # 检查数据文件
    print("\n📂 Checking data files...")
    required_files = [
        (os.path.join(config.data_dir, config.train_file), "Training data"),
        (os.path.join(config.data_dir, config.dev_file), "Development data"),
        (os.path.join(config.data_dir, config.test_file), "Test data"),
        (config.a2c_dict_file, "Article→Charge mapping"),
        (config.c2a_dict_file, "Charge→Article mapping")
    ]

    all_exist = True
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: {file_path} (NOT FOUND)")
            all_exist = False

    if not all_exist:
        print("\n⚠️  Warning: Some required files are missing!")
        print("   Please prepare your data files before running experiments.")
        return False

    print("\n✅ Environment setup completed!")
    return True


def run_training(config: Config, quick_test: bool = False):
    """运行训练"""
    print("\n" + "="*60)
    print("🚀 Starting Training Phase")
    print("="*60)

    if quick_test:
        print("⚡ Quick test mode enabled (reduced epochs)")
        config.num_epochs = 2
        config.eval_interval = 50

    # 运行训练
    train_main()

    print("\n✅ Training completed!")


def run_tta_experiments(config: Config):
    """运行TTA实验"""
    print("\n" + "="*60)
    print("🧪 Starting TTA Experiments")
    print("="*60)

    # 加载数据和模型
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    test_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.test_file),
        tokenizer,
        shuffle=False
    )

    mapping_dict = MappingDictionary(
        config.a2c_dict_file,
        config.c2a_dict_file
    )

    model_path = os.path.join(config.save_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first!")
        return

    # 实验1：标准TTA
    print("\n" + "="*60)
    print("📊 Experiment 1: Standard TTA")
    print("="*60)
    test_with_tta(config, model_path, test_loader, mapping_dict, tta_mode='standard')

    # 实验2：在线TTA
    print("\n" + "="*60)
    print("📊 Experiment 2: Online TTA")
    print("="*60)
    test_with_tta(config, model_path, test_loader, mapping_dict, tta_mode='online')

    # 实验3：自适应TTA
    print("\n" + "="*60)
    print("📊 Experiment 3: Adaptive TTA")
    print("="*60)
    test_with_tta(config, model_path, test_loader, mapping_dict, tta_mode='adaptive')

    print("\n✅ TTA experiments completed!")


def run_ablation_experiments(config: Config):
    """运行消融实验"""
    print("\n" + "="*60)
    print("🔬 Starting Ablation Study")
    print("="*60)

    # 加载数据
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    test_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.test_file),
        tokenizer,
        shuffle=False
    )

    mapping_dict = MappingDictionary(
        config.a2c_dict_file,
        config.c2a_dict_file
    )

    model_path = os.path.join(config.save_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # 运行消融实验
    ablation_study(config, model_path, test_loader, mapping_dict)

    print("\n✅ Ablation study completed!")


def generate_report(config: Config):
    """生成实验报告"""
    print("\n" + "="*60)
    print("📝 Generating Experiment Report")
    print("="*60)

    results_dir = os.path.join(config.save_dir, 'tta_results')

    if not os.path.exists(results_dir):
        print("❌ No results found!")
        return

    # 收集所有结果
    report_data = {
        'config': str(config),
        'experiments': []
    }

    # 读取各个实验结果
    result_files = [
        'tta_results_standard.json',
        'tta_results_online.json',
        'tta_results_adaptive.json',
        'ablation_study.json'
    ]

    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                report_data['experiments'].append({
                    'name': filename.replace('.json', ''),
                    'data': data
                })

    # 保存综合报告
    report_path = os.path.join(config.save_dir, 'experiment_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved to {report_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("📊 Experiment Summary")
    print("="*60)

    for exp in report_data['experiments']:
        print(f"\n{exp['name']}:")
        if 'tta_metrics' in exp['data']:
            metrics = exp['data']['tta_metrics']
            print(f"  Avg Macro-F1: {metrics.get('avg_ma_f1', 0):.4f}")
            print(f"  Alignment Ratio: {metrics.get('alignment_ratio', 0):.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run TTA-LJP experiments')

    parser.add_argument('--mode', type=str, default='full',
                       choices=['setup', 'train', 'test', 'ablation', 'full', 'report'],
                       help='Experiment mode')

    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode with reduced epochs')

    parser.add_argument('--create-dicts', action='store_true',
                       help='Create mapping dictionaries from training data')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')

    # 创新点开关
    parser.add_argument('--no-dynamic-weighting', action='store_true',
                       help='Disable dynamic weighting')
    parser.add_argument('--no-asymmetric', action='store_true',
                       help='Disable asymmetric alignment')
    parser.add_argument('--no-gating', action='store_true',
                       help='Disable confidence gating')
    parser.add_argument('--use-triple', action='store_true',
                       help='Enable triple consistency')

    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 应用命令行参数
    if args.no_dynamic_weighting:
        config.use_dynamic_weighting = False
    if args.no_asymmetric:
        config.use_asymmetric_alignment = False
    if args.no_gating:
        config.use_confidence_gating = False
    if args.use_triple:
        config.use_triple_consistency = True

    # 打印配置
    print("\n" + "="*60)
    print("⚙️  Configuration")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Quick test: {args.quick_test}")
    print(f"\nInnovations:")
    print(f"  Dynamic weighting: {config.use_dynamic_weighting}")
    print(f"  Asymmetric alignment: {config.use_asymmetric_alignment}")
    print(f"  Confidence gating: {config.use_confidence_gating}")
    print(f"  Triple consistency: {config.use_triple_consistency}")
    print("="*60)

    # 执行实验
    try:
        if args.mode == 'setup':
            setup_experiment(config, create_dicts=args.create_dicts)

        elif args.mode == 'train':
            if not setup_experiment(config):
                return
            run_training(config, quick_test=args.quick_test)

        elif args.mode == 'test':
            run_tta_experiments(config)

        elif args.mode == 'ablation':
            run_ablation_experiments(config)

        elif args.mode == 'report':
            generate_report(config)

        elif args.mode == 'full':
            # 完整实验流程
            if not setup_experiment(config, create_dicts=args.create_dicts):
                print("\n❌ Setup failed. Please check your data files.")
                return

            print("\n" + "="*60)
            print("🎯 Running Full Experiment Pipeline")
            print("="*60)
            print("\nSteps:")
            print("  1. Training")
            print("  2. TTA Experiments")
            print("  3. Ablation Study")
            print("  4. Report Generation")
            print("="*60)

            input("\nPress Enter to continue or Ctrl+C to cancel...")

            # 1. 训练
            run_training(config, quick_test=args.quick_test)

            # 2. TTA实验
            run_tta_experiments(config)

            # 3. 消融实验
            run_ablation_experiments(config)

            # 4. 生成报告
            generate_report(config)

            print("\n" + "="*60)
            print("✨ Full experiment pipeline completed!")
            print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
