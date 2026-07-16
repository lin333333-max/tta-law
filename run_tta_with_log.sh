#!/bin/bash
# 运行 TTA 测试并同时保存终端输出到日志文件

if [ $# -eq 0 ]; then
    echo "用法: $0 <mode> [--subset N]"
    echo "示例:"
    echo "  $0 standard"
    echo "  $0 standard --subset 2000"
    echo "  $0 ablation"
    exit 1
fi

MODE=$1
shift  # 移除第一个参数，剩下的作为额外参数

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/tta_runs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/tta_${MODE}_${TIMESTAMP}.log"

echo "🚀 启动 TTA 测试 (模式: $MODE)"
echo "📝 日志文件: $LOG_FILE"
echo "📊 结果文件: ./checkpoints/tta_results/tta_results_${MODE}.json"
echo ""
echo "💡 提示: 可以用 tail -f $LOG_FILE 实时查看进度"
echo ""

# 运行测试，同时输出到终端和日志文件
python test_tta.py --mode "$MODE" "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成！"
else
    echo "❌ 测试出错，退出码: $EXIT_CODE"
    echo "📝 查看完整日志: cat $LOG_FILE"
fi

exit $EXIT_CODE
