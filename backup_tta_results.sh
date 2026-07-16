#!/bin/bash
# 备份 TTA 实验结果，避免被新实验覆盖

RESULTS_DIR="./checkpoints/tta_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${RESULTS_DIR}/backup_${TIMESTAMP}"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 结果目录不存在: $RESULTS_DIR"
    exit 1
fi

# 检查是否有文件需要备份
if [ -z "$(ls -A $RESULTS_DIR/*.json 2>/dev/null)" ]; then
    echo "⚠️  没有找到需要备份的 JSON 文件"
    exit 0
fi

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份所有 JSON 文件
cp $RESULTS_DIR/*.json "$BACKUP_DIR/" 2>/dev/null

echo "✅ 已备份到: $BACKUP_DIR"
echo ""
echo "备份文件列表:"
ls -lh "$BACKUP_DIR"

echo ""
echo "📝 当前配置:"
grep "confidence.*threshold" ./config.py | sed 's/^/  /'
