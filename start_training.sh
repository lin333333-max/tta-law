#!/bin/bash
# 一键启动训练脚本

echo "================================================================"
echo "                  训练启动脚本"
echo "================================================================"

# 1. 清理可能存在的进程
echo ""
echo "Step 1: 清理旧进程..."
pkill -f "python.*train_ultrafast" 2>/dev/null
sleep 2
echo "✓ 已清理"

# 2. 检查磁盘空间
echo ""
echo "Step 2: 检查磁盘空间..."
df -h /root | tail -1
echo ""

# 3. 检查checkpoint
echo "Step 3: 检查checkpoint..."
ls -lh checkpoints/*.pt 2>/dev/null | tail -3
echo ""

# 4. 启动训练
echo "Step 4: 启动训练..."
echo "从 checkpoint_epoch_11.pt 恢复，继续到 epoch 40"
echo ""

cd /root/tta-law
nohup python train_ultrafast.py --resume checkpoints/checkpoint_epoch_11.pt > train_final.log 2>&1 &

PID=$!
echo "✓ 训练已在后台启动 (PID: $PID)"
echo ""

# 5. 等待并显示初始输出
echo "等待训练启动..."
sleep 5
echo ""
echo "最新日志:"
tail -20 train_final.log

echo ""
echo "================================================================"
echo "训练已启动！"
echo "================================================================"
echo ""
echo "监控命令:"
echo "  tail -f train_final.log          # 查看实时日志"
echo "  bash monitor_training.sh         # 运行监控脚本"
echo "  ps aux | grep train_ultrafast    # 检查进程"
echo ""
echo "预计完成时间: ~3.5小时"
echo "目标: Ma-F1 48-52%"
echo "================================================================"
