#!/bin/bash
# 训练监控脚本

echo "================================"
echo "训练状态监控"
echo "================================"

# 检查进程
echo ""
echo "1. 训练进程:"
ps aux | grep "python.*train_ultrafast" | grep -v grep | head -1 || echo "  未找到训练进程"

# 检查磁盘
echo ""
echo "2. 磁盘空间:"
df -h /root | tail -1

# 检查checkpoint
echo ""
echo "3. Checkpoint文件:"
ls -lh /root/tta-law/checkpoints/*.pt 2>/dev/null | wc -l | xargs echo "  当前checkpoint数量:"

# 检查训练历史
echo ""
echo "4. 最新训练进度:"
python << 'PYTHON'
import json
import os
try:
    with open('./logs/training_history.json', 'r') as f:
        history = json.load(f)
    if history:
        latest = history[-1]
        print(f"  Epoch {latest['epoch']}: Ma-F1 = {latest['dev_metrics']['avg_ma_f1']*100:.2f}%")
    else:
        print("  训练历史为空")
except:
    print("  训练历史文件不存在")
PYTHON

echo ""
echo "================================"
