"""
快速测试脚本：验证修复
"""
import sys
import os

print("=" * 60)
print("🔧 测试修复")
print("=" * 60)

# 1. 测试GPU
print("\n1️⃣ 测试GPU...")
try:
    import torch
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU可用 - 将使用GPU训练")
    else:
        print(f"   ⚠️  GPU不可用 - 将使用CPU训练（速度较慢）")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 2. 测试配置
print("\n2️⃣ 测试配置...")
try:
    from config import Config
    config = Config()
    print(f"   设备: {config.device}")
    print(f"   Workers: {config.num_workers}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   ✅ 配置加载成功")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 3. 测试数据加载（如果数据存在）
print("\n3️⃣ 测试数据加载...")
try:
    from transformers import BertTokenizer
    from data_loader import LegalDataset

    train_file = './dataset/train.json'
    if os.path.exists(train_file):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # 创建数据集
        dataset = LegalDataset(
            train_file,
            tokenizer,
            max_length=512,
            num_articles=121,
            num_charges=150
        )

        # 测试获取一个样本
        sample = dataset[0]

        print(f"   样本键: {sample.keys()}")
        print(f"   Input IDs形状: {sample['input_ids'].shape}")
        print(f"   法条标签形状: {sample['articles'].shape}")
        print(f"   罪名标签形状: {sample['charges'].shape}")
        print(f"   法条标签数量: {sample['articles'].sum()}")
        print(f"   罪名标签数量: {sample['charges'].sum()}")
        print(f"   ✅ 数据加载成功 - 类型问题已修复")
    else:
        print(f"   ⚠️  训练文件不存在: {train_file}")
        print(f"   请确保数据文件在正确位置")
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试映射字典（如果存在）
print("\n4️⃣ 测试映射字典...")
try:
    from data_loader import MappingDictionary

    a2c_file = './dataset/article_to_charge.json'
    c2a_file = './dataset/charge_to_article.json'

    if os.path.exists(a2c_file) and os.path.exists(c2a_file):
        mapping_dict = MappingDictionary(a2c_file, c2a_file)

        # 测试映射
        test_article = 234
        mapped_charges = mapping_dict.get_mapped_charges(test_article)
        print(f"   法条 {test_article} 映射到罪名: {mapped_charges}")

        # 测试张量映射
        import torch
        article_probs = torch.rand(2, 121).to(config.device)
        mapped_charges_tensor = mapping_dict.articles_to_charges_tensor(article_probs, config.device)

        print(f"   张量映射输出形状: {mapped_charges_tensor.shape}")
        print(f"   ✅ 映射字典工作正常 - 类型问题已修复")
    else:
        print(f"   ⚠️  映射字典不存在")
        print(f"   请运行: python run_experiment.py --mode setup --create-dicts")
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试模型
print("\n5️⃣ 测试模型...")
try:
    from model import LegalJudgmentModel

    model = LegalJudgmentModel(config).to(config.device)

    # 测试前向传播
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 128)).to(config.device)
    attention_mask = torch.ones(batch_size, 128).to(config.device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   法条预测形状: {output['article_probs'].shape}")
    print(f"   罪名预测形状: {output['charge_probs'].shape}")
    print(f"   ✅ 模型测试成功")
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("📋 总结")
print("=" * 60)
print("\n✅ 主要修复:")
print("   1. 数据加载器类型转换（字符串→整数）")
print("   2. 映射字典类型转换")
print("   3. DataLoader workers根据设备自动调整")
print("   4. GPU自动检测和配置")

print("\n📝 下一步:")
if torch.cuda.is_available():
    print("   ✅ GPU已就绪，可以开始训练:")
    print("      python train.py")
else:
    print("   ⚠️  当前使用CPU，建议:")
    print("      1. 安装CUDA版本的PyTorch:")
    print("         pip uninstall torch")
    print("         pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("      2. 或者使用小批量快速测试:")
    print("         修改config.py: batch_size = 4, num_epochs = 2")

print("\n" + "=" * 60)
