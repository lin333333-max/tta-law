# 法律判决预测的测试时适应 (TTA-LJP)

基于 K-LJP 和 PCL 融合的法律判决预测系统，集成创新的测试时适应技术。

## 🌟 创新点

本项目实现了以下5个创新点：

### ✅ 创新点1：动态加权机制
- 基于样本级对齐度动态调整损失权重
- 高对齐度样本降低更新强度，避免过度优化
- 低对齐度样本加大跨任务修复力度

### ✅ 创新点2：不对称更新策略
- 基于K-LJP消融实验发现，罪名预测更可靠
- 让罪名预测更多地引导法条预测
- 使用非对称KL散度约束（α=0.7, β=0.3）

### ✅ 创新点3：置信度门控
- 保护高置信度样本，避免改对为错
- 只对低置信度样本执行TTA更新
- 一致性门控：检测"高置信度+低对齐度"的危险信号

### ⚠️ 创新点4：动态对齐字典（监控模式）
- 记录字典外的新映射组合
- 作为监控功能，供专家审核

### ✅ 创新点5：三重一致性（可选）
- 任务内特征扰动一致性（PCL）
- 任务内文本语义扰动一致性
- 任务间知识对齐一致性（K-LJP）

## 📁 项目结构

```
tta-law/
├── config.py              # 配置文件
├── model.py               # K-LJP模型架构
├── data_loader.py         # 数据加载和映射字典
├── loss_functions.py      # 创新点实现（动态加权、不对称更新、置信度门控）
├── tta_trainer.py         # TTA训练器
├── evaluator.py           # 评估和错误分析
├── train.py               # 主训练脚本
├── test_tta.py            # TTA测试脚本
├── run_experiment.py      # 快速实验脚本
├── requirements.txt       # 依赖包
└── README.md             # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

数据格式（CAIL2018）：
```json
{
  "fact": "案件事实描述...",
  "meta": {
    "relevant_articles": [234, 343],
    "accusation": [0, 8, 15]
  }
}
```

数据文件结构：
```
dataset/
├── train.json              # 训练集
├── dev.json                # 验证集
├── test.json               # 测试集
├── article_to_charge.json  # 法条→罪名映射
└── charge_to_article.json  # 罪名→法条映射
```

### 3. 构建映射字典（首次运行）

```python
from data_loader import create_mapping_dict_from_cail

create_mapping_dict_from_cail(
    './dataset/train.json',
    './dataset/article_to_charge.json',
    './dataset/charge_to_article.json'
)
```

### 4. 训练模型

```bash
# 基础训练（不使用三重一致性）
python train.py

# 使用三重一致性训练
python train.py --use-triple-consistency
```

### 5. TTA测试

```bash
# 标准TTA测试
python test_tta.py --mode standard

# 在线TTA测试（状态累积）
python test_tta.py --mode online

# 自适应TTA测试（动态步数）
python test_tta.py --mode adaptive

# 消融实验
python test_tta.py --mode ablation
```

## ⚙️ 配置说明

在 `config.py` 中可以调整以下关键参数：

### 创新点开关
```python
# 创新点1：动态加权
use_dynamic_weighting = True
base_lambda = 1.0
sensitivity = 2.0

# 创新点2：不对称更新
use_asymmetric_alignment = True
alpha = 0.7  # 罪名→法条权重
beta = 0.3   # 法条→罪名权重

# 创新点3：置信度门控
use_confidence_gating = True
confidence_high_threshold = 0.7
confidence_low_threshold = 0.4

# 创新点5：三重一致性
use_triple_consistency = False
```

### TTA参数
```python
tta_learning_rate = 1e-4
tta_steps = 5
```

## 📊 实验结果

### 基线 vs TTA 对比

| 指标 | Baseline | TTA | 提升 |
|------|----------|-----|------|
| Article Macro-F1 | - | - | - |
| Charge Macro-F1 | - | - | - |
| Avg Macro-F1 | - | - | - |
| Alignment Ratio | - | - | - |

### 消融实验

| 配置 | Avg Macro-F1 | 提升 |
|------|--------------|------|
| 无创新点 (基线) | - | - |
| +不对称更新 | - | - |
| +置信度门控 | - | - |
| +动态加权 | - | - |
| 全部创新点 | - | - |

## 🔍 核心实现细节

### 1. 动态加权计算

```python
# 计算样本级对齐度
align_score = compute_sample_alignment(article_probs, charge_probs, mapping_dict)

# 动态λ：对齐度低→λ大
lambda_dynamic = base_lambda * sigmoid(sensitivity * (0.5 - align_score))

# 应用动态权重
L_total = L_ce + lambda_dynamic * L_align
```

### 2. 不对称对齐损失

```python
# 罪名→法条：强约束（α=0.7）
mapped_articles = charges_to_articles(charge_probs.detach())
L_c2a = KL(article_probs || mapped_articles)

# 法条→罪名：弱约束（β=0.3）
mapped_charges = articles_to_charges(article_probs)
L_a2c = KL(charge_probs || mapped_charges)

L_align = alpha * L_c2a + beta * L_a2c
```

### 3. 置信度门控

```python
# 计算置信度
confidence = compute_confidence(article_probs, charge_probs)

# 门控规则
if confidence > threshold_high:
    skip_update()  # 保护高置信度
elif confidence < threshold_low:
    if not dangerous_pattern(align_score):
        do_tta_update()  # 只更新安全的低置信度样本
```

## 📈 日志和可视化

训练过程会自动记录到TensorBoard：

```bash
tensorboard --logdir ./logs
```

查看内容：
- 训练损失曲线
- 各子损失项（CE、Align、PCL）
- 学习率变化
- 验证集指标

## 🔧 自定义扩展

### 添加新的扰动策略

在 `model.py` 中修改 `forward_with_feature_perturbation`：

```python
def forward_with_feature_perturbation(self, ...):
    # 添加你的扰动方法
    custom_perturbation = your_perturbation_function(fact_hidden)
    # ...
```

### 实现文本语义扰动（创新点5）

在 `loss_functions.py` 中实现：

```python
def semantic_perturbation(self, text):
    # 方法1：同义词替换
    perturbed_text = synonym_replacement(text)
    
    # 方法2：回译
    # perturbed_text = back_translation(text)
    
    # 方法3：LLM改写
    # perturbed_text = llm_paraphrase(text)
    
    return perturbed_text
```

## 📝 论文引用

本项目基于以下论文：

1. **K-LJP**: Legal Judgment Prediction based on Knowledge-enhanced Multi-Task and Multi-Label Text Classification (NAACL 2025)

2. **PCL**: Test-Time Adaptation with Perturbation Consistency Learning (arXiv 2023)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- CAIL2018数据集
- Hugging Face Transformers
- PyTorch团队

## 📧 联系

如有问题，请提交Issue或联系 [your-email@example.com]

---

**注意事项：**
1. 确保GPU内存足够（建议≥16GB）
2. 首次运行需要下载预训练模型（约400MB）
3. TTA过程会显著增加推理时间（约2-10倍）
4. 法律场景建议启用置信度门控，避免高风险错误
