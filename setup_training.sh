#!/bin/bash
# 完整的训练准备脚本

echo "================================================================"
echo "           重新训练准备脚本"
echo "================================================================"

echo ""
echo "Step 1: 使用对齐后的数据..."
cp dataset/train_kljp_aligned.json dataset/train.json
cp dataset/dev_kljp_aligned.json dataset/dev.json
cp dataset/test_kljp_aligned.json dataset/test.json
echo "✓ 数据文件已更新"

echo ""
echo "Step 2: 优化config.py..."
python << 'PYTHON'
import re

with open('config.py', 'r') as f:
    content = f.read()

# 更新标签数
content = re.sub(r'num_articles = \d+', 'num_articles = 114', content)
content = re.sub(r'num_charges = \d+', 'num_charges = 141', content)

# 优化训练超参数
content = re.sub(r'learning_rate = 2e-5', 'learning_rate = 5e-6', content)  # 降低到5e-6
content = re.sub(r'num_epochs = \d+', 'num_epochs = 40', content)  # 增加到40
content = re.sub(r'batch_size = 32', 'batch_size = 16', content)  # 减小batch
content = re.sub(r'warmup_ratio = 0\.1', 'warmup_ratio = 0.15', content)  # 增加warmup

# 增强对齐损失
content = re.sub(r'base_lambda = [0-9.]+', 'base_lambda = 3.0', content)  # 增大到3.0

# 优化TTA参数
content = re.sub(r'tta_learning_rate = 1e-4', 'tta_learning_rate = 5e-5', content)

with open('config.py', 'w') as f:
    f.write(content)

print("✓ config.py已优化:")
print("  - num_articles = 114")
print("  - num_charges = 141")
print("  - learning_rate = 5e-6 (更小)")
print("  - num_epochs = 40 (更多)")
print("  - batch_size = 16 (更小，对罕见标签友好)")
print("  - warmup_ratio = 0.15")
print("  - base_lambda = 3.0 (更强对齐)")
PYTHON

echo ""
echo "Step 3: 重建映射字典..."
python build_mappings.py | tail -5

echo ""
echo "Step 4: 清理旧文件..."
rm -rf dataset/cache/*
rm -f checkpoints/checkpoint_epoch_*.pt
echo "✓ 缓存和旧checkpoint已清理"

echo ""
echo "Step 5: 重新预处理数据..."
python preprocess_data.py

echo ""
echo "================================================================"
echo "✓ 准备完成！现在可以开始训练:"
echo "  python train_ultrafast.py"
echo ""
echo "或在tmux中训练:"
echo "  tmux new -s train"
echo "  python train_ultrafast.py"
echo "================================================================"
