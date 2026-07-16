#!/bin/bash
# 项目状态检查脚本

echo "================================================================================"
echo "                        项目状态检查报告"
echo "================================================================================"
echo ""

echo "1. 已完成的工作"
echo "--------------------------------------------------------------------------------"
echo "✓ 阶段1: 环境搭建完成"
echo "✓ 阶段2: 基础训练完成 (20 epochs)"
echo "✓ 阶段3: TTA测试完成 (3种模式)"
echo "✓ 断点续跑功能已添加到 train_ultrafast.py"
echo ""

echo "2. 训练结果"
echo "--------------------------------------------------------------------------------"
if [ -f "./checkpoints/best_model.pt" ]; then
    echo "✓ 最佳模型: ./checkpoints/best_model.pt (Epoch 18)"
else
    echo "✗ 最佳模型不存在"
fi

if [ -f "./logs/training_history.json" ]; then
    best_epoch=$(cat ./logs/training_history.json | grep -o '"epoch": [0-9]*' | tail -1 | grep -o '[0-9]*')
    echo "✓ 训练历史: 共 $best_epoch epochs"
else
    echo "✗ 训练历史不存在"
fi

echo ""
echo "测试集性能:"
if [ -f "./checkpoints/test_results_ultrafast.json" ]; then
    avg_maf1=$(cat ./checkpoints/test_results_ultrafast.json | grep "avg_ma_f1" | grep -o '[0-9.]*' | head -1)
    echo "  Avg Ma-F1: ${avg_maf1}% (目标: 60%+)"
fi

echo ""

echo "3. TTA测试结果"
echo "--------------------------------------------------------------------------------"
for mode in standard online adaptive; do
    file="./checkpoints/tta_results/tta_results_${mode}.json"
    if [ -f "$file" ]; then
        echo "✓ ${mode} 模式: 已完成"
    else
        echo "✗ ${mode} 模式: 未完成"
    fi
done
echo ""

echo "4. 核心问题"
echo "--------------------------------------------------------------------------------"
echo "问题1: Ma-F1只有30%，远低于论文60%"
echo "问题2: 类别极度不平衡 (16,225:1)"
echo "问题3: 罕见标签预测崩溃"
echo ""

echo "5. 断点续跑功能"
echo "--------------------------------------------------------------------------------"
if grep -q "resume_from" ./train_ultrafast.py; then
    echo "✓ train_ultrafast.py 已支持断点续跑"
    echo ""
    echo "使用方法:"
    echo "  python train_ultrafast.py --resume ./checkpoints/checkpoint_epoch_10.pt"
else
    echo "✗ 断点续跑功能未添加"
fi
echo ""

echo "6. 下一步行动"
echo "--------------------------------------------------------------------------------"
echo "立即执行（方案A - 快速验证）:"
echo "  1. 编辑 config.py，改3行:"
echo "     learning_rate = 1e-5"
echo "     num_epochs = 30"
echo "     base_lambda = 1.5"
echo ""
echo "  2. 重新训练:"
echo "     python train_ultrafast.py"
echo ""
echo "  3. 观察Ma-F1是否提升到35%+"
echo ""
echo "完整优化（方案B - 效果最好）:"
echo "  1. 实施Focal Loss (修改loss_functions.py)"
echo "  2. 修改config.py"
echo "  3. 重新训练"
echo ""

echo "7. 可用的分析工具"
echo "--------------------------------------------------------------------------------"
echo "  python analyze_label_distribution.py  - 分析标签分布"
echo "  python analyze_tta_results.py         - 分析TTA结果"
echo "  python evaluate_multilabel_only.py    - 评估多标签样本"
echo ""

echo "8. 文档"
echo "--------------------------------------------------------------------------------"
echo "  QUICK_SUMMARY.txt   - 快速总结 (必读!)"
echo "  ACTION_GUIDE.md     - 完整行动指南"
echo "  PROJECT_SUMMARY.md  - 项目总结"
echo ""

echo "================================================================================"
echo "状态检查完成！建议先阅读 QUICK_SUMMARY.txt"
echo "================================================================================"
