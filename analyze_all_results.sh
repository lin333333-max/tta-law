#!/bin/bash
# TTA分析与优化建议脚本

echo "======================================================================"
echo "TTA测试结果分析与优化建议"
echo "======================================================================"

# 1. 分析TTA结果
echo ""
echo "1. 分析TTA三种模式结果..."
python analyze_tta_results.py

# 2. 对比不同评估结果
echo ""
echo "======================================================================"
echo "2. 对比不同评估结果"
echo "======================================================================"

echo ""
echo "评估结果对比："
echo "  完整测试集评估 (test_results_ultrafast.json):"
cat checkpoints/test_results_ultrafast.json | grep -A 3 "avg_ma_f1"

echo ""
echo "  多标签样本评估 (multilabel_only_evaluation.json):"
cat checkpoints/multilabel_only_evaluation.json | grep -A 3 "avg_ma_f1"

echo ""
echo "  TTA Standard模式 Baseline:"
cat checkpoints/tta_results/tta_results_standard.json | grep -A 1 '"baseline_metrics"' | grep "avg_ma_f1"

echo ""
echo "  TTA Adaptive模式结果:"
cat checkpoints/tta_results/tta_results_adaptive.json | grep -A 1 '"tta_metrics"' | grep "avg_ma_f1"

echo ""
echo "======================================================================"
echo "3. 关键发现总结"
echo "======================================================================"

echo ""
echo "✓ 已完成的工作:"
echo "  - 基础训练: 20 epochs, Best Ma-F1 = 31.58% (Epoch 18)"
echo "  - 完整测试集评估: Avg Ma-F1 = 30.86%"
echo "  - 多标签样本评估: Avg Ma-F1 = 28.48%"
echo "  - TTA三种模式测试: Standard, Online, Adaptive"
echo ""
echo "⚠️  核心问题:"
echo "  1. 类别极度不平衡: 法条16,225:1, 罪名9,198:1"
echo "  2. Ma-F1远低于论文: 28-30% vs 60% (差距30%)"
echo "  3. TTA Baseline异常: 只有7.74% Ma-F1"
echo ""
echo "💡 优化策略:"
echo "  优先级1: 解决类别不平衡 (Focal Loss + 类别权重)"
echo "  优先级2: 调整超参数 (LR=1e-5, Epochs=30)"
echo "  优先级3: 数据增强 (针对罕见标签)"
echo "  优先级4: TTA优化 (在基础模型好的情况下)"

echo ""
echo "======================================================================"
echo "4. 下一步建议"
echo "======================================================================"

echo ""
echo "立即执行:"
echo "  1. 修改config.py: learning_rate=1e-5, num_epochs=30"
echo "  2. 实施Focal Loss解决类别不平衡"
echo "  3. 使用断点续跑重新训练:"
echo "     python train_resumable.py"
echo ""
echo "预期效果:"
echo "  阶段2优化后: 30% → 45% Ma-F1 (+15%)"
echo "  阶段3优化后: 45% → 55% Ma-F1 (+10%)"
echo "  阶段4优化后: 55% → 60% Ma-F1 (+5%)"
echo "  达成目标: 超过K-LJP论文60% ✓"

echo ""
echo "======================================================================"
echo "分析完成！详细文档见: TTA_ANALYSIS_AND_RESUME_GUIDE.md"
echo "======================================================================"
