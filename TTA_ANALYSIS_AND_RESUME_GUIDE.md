# TTA测试结果分析 & 断点续跑功能说明

## 📊 TTA测试结果总结

### 一、三种模式性能对比

```
模式          Baseline    Standard    Online      Adaptive    最佳
------------------------------------------------------------------------
Avg Ma-F1     7.74%      30.83%      3.48%       30.96%      Adaptive ✓
Avg Mi-F1     79.25%     57.47%      7.17%       57.57%      Baseline ✓
Alignment     93.40%     84.68%      86.66%      85.49%      Baseline ✓
更新率        N/A        0%          2.0%        18.4%       Adaptive
```

### 二、核心发现

#### 🔴 异常问题
**Baseline的Ma-F1只有7.74%，远低于之前评估的30.86%**

可能原因：
1. TTA测试使用了不同的batch_size（8 vs 64）
2. 不同的数据加载方式
3. 可能使用了不同的评估子集

**结论**：这个Baseline不可靠，需要重新评估！

#### ✅ TTA效果（相对异常Baseline）
- **Ma-F1提升**: 7.74% → 30.96% (+300%)
- **Mi-F1下降**: 79.25% → 57.57% (-27%)

**矛盾现象**：罕见标签变好，但常见标签变差

#### 💡 真实TTA价值评估
假设真实Baseline = 30% Ma-F1（与之前一致）：
- TTA预期提升：30% → 31-33%
- 提升幅度：3-10%
- **仍然远低于K-LJP论文的60%**

### 三、模式对比分析

#### Standard模式（所有创新点关闭）
```
配置:
  use_dynamic_weighting: False
  use_asymmetric_alignment: False
  use_confidence_gating: False
  
结果: 30.83% Ma-F1
更新率: 0% (无更新)
结论: 等同于Baseline
```

#### Online模式（状态累积）
```
更新率: 2.0% (658/32508)
结果: 3.48% Ma-F1
结论: 更新率太低，几乎崩溃
原因: 置信度门控太严格，大部分样本被跳过
```

#### Adaptive模式（动态步数）✓ 推荐
```
更新率: 18.4% (3587/19540)
结果: 30.96% Ma-F1
结论: 性能最佳，更新率适中
机制: 根据置信度动态调整更新步数
```

### 四、关键结论

#### 1. TTA的真实作用有限
- 在基础模型Ma-F1只有30%时，TTA最多提升到33%
- **无法弥补与K-LJP论文60%的30%差距**
- TTA是"锦上添花"，不是"雪中送炭"

#### 2. 应该优先优化阶段2（基础训练）
```
当前优先级:
  1. 解决类别不平衡问题（Focal Loss / 类别权重）  ← 最重要
  2. 调整超参数（学习率、epochs、阈值）
  3. 数据增强（针对罕见标签）
  4. 改进模型架构
  5. TTA优化（最后）
```

#### 3. 立即行动建议
```
☑️ 已完成:
  ✓ TTA三种模式测试
  ✓ 识别出Baseline评估异常
  ✓ 确认TTA提升有限

⚠️ 待执行:
  1. 重新用相同配置评估真实Baseline
  2. 实施Focal Loss解决类别不平衡
  3. 降低学习率到1e-5
  4. 增加训练轮数到30-40 epochs
  5. 搜索最佳预测阈值
```

---

## 🔄 断点续跑功能使用说明

### 功能特性

新的训练脚本 `train_resumable.py` 支持：
- ✅ 从任意checkpoint恢复训练
- ✅ 保存完整训练状态（模型、优化器、scheduler、scaler）
- ✅ 自动恢复训练历史
- ✅ 继续使用之前的best_metric
- ✅ 全局步数（global_step）不重置

### 使用方法

#### 1. 从头开始训练
```bash
python train_resumable.py
```

#### 2. 从checkpoint恢复训练
```bash
# 从Epoch 10恢复
python train_resumable.py --resume ./checkpoints/checkpoint_epoch_10.pt

# 从最佳模型恢复
python train_resumable.py --resume ./checkpoints/best_model.pt
```

#### 3. 只评估，不训练
```bash
python train_resumable.py --test-only
```

### 断点续跑示例

假设训练中断在Epoch 15：

```bash
# 训练自动保存的checkpoint
ls checkpoints/
  checkpoint_epoch_5.pt
  checkpoint_epoch_10.pt
  checkpoint_epoch_15.pt
  best_model.pt

# 从Epoch 15恢复，继续训练到Epoch 30
python train_resumable.py --resume ./checkpoints/checkpoint_epoch_15.pt
```

恢复后会显示：
```
============================================================
从checkpoint恢复训练: ./checkpoints/checkpoint_epoch_15.pt
============================================================
✓ 模型权重已恢复
✓ 训练状态已恢复:
  - 起始Epoch: 16
  - Global Step: 23100
  - Best Metric: 0.3158
✓ 训练历史已恢复 (15 epochs)
✓ 优化器状态已恢复
✓ Scheduler状态已恢复
✓ GradScaler状态已恢复
============================================================
```

### Checkpoint包含内容

每个checkpoint文件保存：
```python
{
  'epoch': 当前轮数,
  'global_step': 全局步数,
  'model_state_dict': 模型权重,
  'optimizer_state_dict': 优化器状态,
  'scheduler_state_dict': 学习率调度器状态,
  'scaler_state_dict': 混合精度scaler状态,
  'best_metric': 历史最佳指标,
  'config': 配置对象
}
```

### 注意事项

1. **断点续跑会保持所有训练状态**
   - 学习率会从中断处继续调度
   - 优化器动量等状态完整保留
   - 训练历史JSON文件会自动续写

2. **checkpoint文件较大（~608MB）**
   - 包含完整的BERT模型权重
   - 包含优化器状态（Adam需要为每个参数保存动量）
   - 定期清理不需要的旧checkpoint

3. **修改config后不能直接恢复**
   - 如果改变了模型结构（如num_decoder_layers）
   - 需要从头训练，或只加载模型权重

---

## 📋 下一步行动计划

### 阶段1: 确认真实Baseline（立即执行）
```bash
# 重新评估，确保使用相同配置
python evaluate_multilabel_only.py
```

### 阶段2: 优化基础模型（优先级最高）

#### 2.1 实施Focal Loss
修改 `loss_functions.py`，添加类别平衡损失

#### 2.2 调整超参数
修改 `config.py`:
```python
learning_rate = 1e-5       # 从2e-5降低
num_epochs = 30            # 从20增加
base_lambda = 1.5          # 增强对齐约束
```

#### 2.3 重新训练
```bash
python train_resumable.py
```

预期效果: Ma-F1 从30% → 40-45%

### 阶段3: 数据增强与高级优化

#### 3.1 类别权重采样
#### 3.2 针对罕见标签的数据增强
#### 3.3 优化预测阈值

预期效果: Ma-F1 从40-45% → 50-55%

### 阶段4: TTA优化（最后）

在基础模型达到50%+ Ma-F1后，TTA才能发挥价值：
```bash
# 运行消融实验
python test_tta.py --mode ablation
```

预期效果: Ma-F1 从50-55% → 55-60%

---

## 🎯 目标路线图

```
当前状态: 30% Ma-F1
    ↓
阶段2优化: Focal Loss + 超参数调优
    → 预期: 40-45% Ma-F1 (+10-15%)
    ↓
阶段3优化: 数据增强 + 类别采样
    → 预期: 50-55% Ma-F1 (+10%)
    ↓
阶段4优化: TTA + 模型架构
    → 预期: 58-62% Ma-F1 (+8-10%)
    ↓
目标达成: 超过K-LJP论文60% ✓
```

---

## 📞 问题诊断清单

遇到问题时，检查：

- [ ] Baseline评估是否使用了相同的batch_size？
- [ ] 是否在完整测试集上评估（而非subset）？
- [ ] 训练是否收敛（loss不再下降）？
- [ ] 是否使用了缓存的预处理数据？
- [ ] 类别不平衡是否已解决？
- [ ] 学习率是否过大导致震荡？
- [ ] 预测阈值是否合适（默认0.5可能不是最优）？

---

生成时间: 2026-07-16
最后更新: TTA三种模式测试完成
