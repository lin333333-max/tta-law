# 超参数分析与优化建议

## 📊 当前超参数配置

### 训练超参数
```python
batch_size = 32
learning_rate = 2e-5
num_epochs = 20
weight_decay = 0.01
warmup_ratio = 0.1
max_grad_norm = 1.0
```

### TTA超参数
```python
tta_batch_size = 8
tta_learning_rate = 1e-4
tta_steps = 1
```

---

## 🔍 详细分析

### 1. Learning Rate = 2e-5

**当前设置**：2e-5（BERT常用默认值）

**分析**：
- ✅ 这是BERT微调的标准学习率
- ✅ 对于大数据集（>100k样本）效果好
- ⚠️ 但你现在只有39k训练样本（对齐后）
- ⚠️ 数据量减少了76%（从163k → 39k）

**问题**：
```
学习率过大的风险：
  - 样本少，每个epoch看到的数据少
  - 容易在罕见标签上过拟合
  - 可能错过最优解
```

**证据（从训练历史看）**：
```
你之前的训练曲线（未对齐，49k样本）：
  Epoch 18: Ma-F1 = 31.58% (最佳)
  Epoch 20: Ma-F1 = 30.24% (过拟合)
  
说明：2e-5在18轮后就开始过拟合
```

**建议**：
```python
learning_rate = 1e-5  # 降低50%

原因：
  1. 样本量少，需要更小的学习率
  2. 罕见标签需要更细致的学习
  3. K-LJP论文可能也用了更小的学习率
  
预期效果：
  - 收敛更稳定
  - 罕见标签学得更好
  - Ma-F1 预期 +2-5%
```

---

### 2. Batch Size = 32

**当前设置**：32

**分析**：
- ✅ 对于你的GPU（24GB）合理
- ✅ 不会OOM
- ⚠️ 但可能不是最优

**理论**：
```
Batch Size对多标签分类的影响：

小Batch (8-16):
  优点: 梯度噪声大，容易跳出局部最优
        对罕见标签有帮助（每个batch更可能包含）
  缺点: 训练慢，梯度不稳定

中Batch (32-64):
  优点: 平衡速度和稳定性
  缺点: 可能错过罕见标签

大Batch (128+):
  优点: 训练快，梯度稳定
  缺点: 容易卡在局部最优
        罕见标签更难学
```

**你的情况**：
```
数据集特点：
  - 39k训练样本
  - 116法条、141罪名
  - 仍有类别不平衡（虽然已改善）
  
每个epoch的步数：
  39,276 / 32 = 1,227 步/epoch
  20 epochs = 24,540 总步数
```

**建议1：保持32（推荐）**
```python
batch_size = 32  # 不变

原因：
  - 已经是不错的平衡点
  - 训练速度合理
  - 显存利用充分
```

**建议2：尝试16（如果时间允许）**
```python
batch_size = 16  # 减半

原因：
  - 对罕见标签更友好
  - 梯度噪声帮助泛化
  
缺点：
  - 训练时间翻倍
  
预期效果：
  - Ma-F1 可能 +1-3%
```

---

### 3. Num Epochs = 20

**当前设置**：20 epochs

**分析**：
```
你之前的训练（49k样本）：
  Epoch 18: 最佳
  Epoch 20: 开始过拟合
  
现在（39k样本）：
  样本更少，可能更快过拟合
  或者：因为类别平衡了，可能需要更多轮
```

**建议**：
```python
num_epochs = 30  # 增加到30

原因：
  1. 降低学习率后，需要更多epoch收敛
  2. 罕见标签需要更多训练
  3. 使用Early Stopping防止过拟合
  
配合策略：
  - 每5 epoch保存checkpoint
  - 监控dev Ma-F1
  - 如果连续5 epoch不提升就停止
```

---

### 4. Weight Decay = 0.01

**当前设置**：0.01

**分析**：
- ✅ BERT微调的标准值
- ✅ 防止过拟合

**建议**：保持不变
```python
weight_decay = 0.01  # 不变
```

---

### 5. Warmup Ratio = 0.1

**当前设置**：0.1（前10%步数warmup）

**分析**：
```
总步数：39,276 / 32 × 20 = 24,540 步
Warmup步数：2,454 步
```

**建议**：保持或略微增加
```python
warmup_ratio = 0.1  # 保持
# 或
warmup_ratio = 0.15  # 如果增加到30 epochs

原因：
  - 更长的warmup对稳定训练有帮助
  - 特别是降低学习率后
```

---

## 🎯 推荐的超参数配置

### 配置A：保守优化（推荐⭐⭐⭐⭐⭐）

```python
# 主要改动
learning_rate = 1e-5      # 从2e-5降低
num_epochs = 30           # 从20增加
batch_size = 32           # 保持

# 其他保持不变
weight_decay = 0.01
warmup_ratio = 0.1
max_grad_norm = 1.0
```

**预期效果**：
- Ma-F1: 45-50% → 48-53%（+3-5%）
- 训练时间：1.5天（增加50%）
- 风险：低

**理由**：
1. 学习率降低是必要的（样本少了76%）
2. Epochs增加防止欠拟合
3. Batch size保持，训练速度可接受

---

### 配置B：激进优化（如果时间充足）

```python
# 主要改动
learning_rate = 1e-5      # 降低
num_epochs = 40           # 大幅增加
batch_size = 16           # 减半

# 调整
warmup_ratio = 0.15       # 增加warmup
```

**预期效果**：
- Ma-F1: 45-50% → 50-55%（+5-8%）
- 训练时间：4天（翻倍）
- 风险：中等

**理由**：
1. 小batch对罕见标签更友好
2. 更多epochs充分训练
3. 更长warmup保证稳定

---

### 配置C：快速验证（如果急于看结果）

```python
# 主要改动
learning_rate = 1e-5      # 降低
num_epochs = 25           # 略微增加
batch_size = 32           # 保持

# 其他保持不变
```

**预期效果**：
- Ma-F1: 45-50% → 47-52%（+2-4%）
- 训练时间：1.25天
- 风险：低

---

## 📊 学习率调度策略

### 当前策略：Linear Warmup

**建议改进**：使用Cosine Annealing with Warmup

```python
# 在train_ultrafast.py中修改
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**优点**：
- 后期学习率逐渐衰减
- 帮助收敛到更好的解
- 对罕见标签更友好

**预期提升**：+1-2% Ma-F1

---

## 🔬 如何验证超参数是否最优？

### 方法1：学习率范围测试（推荐）

```bash
# 测试3个学习率，每个训练5 epochs
python train_ultrafast.py --lr 5e-6 --epochs 5
python train_ultrafast.py --lr 1e-5 --epochs 5
python train_ultrafast.py --lr 2e-5 --epochs 5

# 观察哪个Ma-F1最高，然后用那个继续训练
```

### 方法2：观察训练曲线

**良好的训练曲线**：
```
Epoch 1-5:   Ma-F1快速上升（学习率合适）
Epoch 6-15:  Ma-F1稳定上升（学习中）
Epoch 16-25: Ma-F1缓慢上升（接近收敛）
Epoch 26+:   Ma-F1稳定或略降（过拟合信号）
```

**学习率过大的信号**：
```
- Ma-F1震荡剧烈
- Dev loss不下降
- 早期过拟合（<10 epochs）
```

**学习率过小的信号**：
```
- Ma-F1上升极慢
- 20 epochs后仍在明显上升
- Loss下降缓慢
```

---

## 💡 K-LJP论文使用的超参数

**论文中提到**：
```
从论文的消融实验和训练描述推测：
- Learning rate: 可能是1e-5或5e-6（未明说）
- Batch size: 可能是16-32
- Epochs: 可能是30-50（直到收敛）
- 使用了Early Stopping
```

**证据**：
```
1. 论文训练集更大（163k vs 你的39k）
2. 但标签数相近（121,150 vs 你的116,141）
3. 达到60% Ma-F1说明训练充分
4. 很可能用了更小的学习率+更多epochs
```

---

## 🎯 最终推荐

### 立即执行（配置A）

```python
# 修改config.py
learning_rate = 1e-5  # ← 改这里
num_epochs = 30       # ← 改这里
batch_size = 32       # 保持
```

**理由**：
1. ✅ 风险最低
2. ✅ 预期效果明显（+3-5%）
3. ✅ 时间成本可接受（1.5天）
4. ✅ 与论文设置更接近

**执行**：
```bash
# 1. 备份当前config
cp config.py config.py.backup

# 2. 修改config.py
python -c "
import re
with open('config.py', 'r') as f:
    content = f.read()
content = re.sub(r'learning_rate = 2e-5', 'learning_rate = 1e-5', content)
content = re.sub(r'num_epochs = 20', 'num_epochs = 30', content)
with open('config.py', 'w') as f:
    f.write(content)
print('✓ 已更新: learning_rate=1e-5, num_epochs=30')
"

# 3. 开始训练
python train_ultrafast.py
```

---

## 📈 预期效果总结

```
当前设置（2e-5, batch=32, epochs=20）：
  Ma-F1 预期: 45-50%

优化后（1e-5, batch=32, epochs=30）：
  Ma-F1 预期: 48-53% (+3-5%)

如果再用配置B（1e-5, batch=16, epochs=40）：
  Ma-F1 预期: 50-55% (+5-8%)

总路线图：
  当前: 28%
  → 对齐: 45-50% (+17-22%)
  → 优化超参: 48-53% (+3-5%)
  → 标签知识: 53-58% (+5%)
  → Focal Loss: 58-62% (+5%)
  目标: 60%+ ✓
```

---

生成时间: 2026-07-16
建议: 使用配置A（learning_rate=1e-5, epochs=30）
