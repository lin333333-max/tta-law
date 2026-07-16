"""
配置文件：包含所有超参数和路径设置
"""
import torch

class Config:
    # ============ 基础设置 ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # 根据设备类型自动设置workers
    _is_cuda = torch.cuda.is_available()

    # ============ 数据路径 ============
    data_dir = './dataset'
    train_file = 'train.json'
    dev_file = 'dev.json'
    test_file = 'test.json'

    # 标签映射文件
    label_mappings_file = './dataset/label_mappings.json'

    # 法条-罪名映射字典路径（使用索引版本）
    a2c_dict_file = './dataset/article_to_charge_indexed.json'  # 法条→罪名
    c2a_dict_file = './dataset/charge_to_article_indexed.json'  # 罪名→法条

    # ============ 模型设置 ============
    # Encoder配置
    encoder_type = 'bert'  # 'bert', 'cnn', 'lstm'
    pretrained_model = '/root/models/bert-base-chinese/models/tiansz--bert-base-chinese/snapshots/master'  # 使用本地模型
    max_seq_length = 512
    hidden_dim = 768

    # Decoder配置（用于标签关系学习）
    num_decoder_layers = 3
    num_attention_heads = 8
    decoder_dropout = 0.1

    # 标签数量（根据实际数据集）
    num_articles = 114  # 法条数量
    num_charges = 141   # 罪名数量

    # ============ 训练设置 ============
    batch_size = 16  # 提升到32（显存充足，从10.8GB/24GB可以看出还有很大空间）
    num_epochs = 40
    learning_rate = 5e-6
    weight_decay = 0.01
    warmup_ratio = 0.15
    max_grad_norm = 1.0

    # ============ TTA设置 ============
    tta_learning_rate = 5e-5
    tta_steps = 1  # PCL论文强调少步数更新，5步→1步可直接提速5倍且更稳定
    tta_batch_size = 8  # TTA需要反向传播+保存优化器状态，显存开销远大于纯推理，需单独设小
    tta_update_layernorm_only = True  # 参考PCL论文：只更新LayerNorm参数，而非全参数更新

    # ============ 创新点1：动态加权 ============
    use_dynamic_weighting = True
    base_lambda = 3.0
    sensitivity = 2.0  # 对齐度敏感度参数

    # ============ 创新点2：不对称更新 ============
    use_asymmetric_alignment = True
    alpha = 0.7  # 罪名→法条的权重（更高，因为罪名更可靠）
    beta = 0.3   # 法条→罪名的权重
    freeze_charge_in_tta = False  # 是否在TTA时完全冻结罪名预测头

    # ============ 创新点3：置信度门控 ============
    use_confidence_gating = True
    confidence_high_threshold = 0.85  # 高置信度阈值（>此值不更新）— 放宽让更多样本参与TTA
    confidence_low_threshold = 0.6    # 低置信度阈值（<此值才更新）— 提高到更合理的范围
    use_consistency_gating = True     # 是否使用一致性门控
    consistency_threshold = 0.3       # 低对齐度阈值

    # ============ 创新点5：三重一致性 ============
    use_triple_consistency = False  # 是否启用三重一致性
    lambda_feature_perturbation = 1.0   # 特征扰动一致性权重
    lambda_semantic_perturbation = 0.5  # 文本语义一致性权重
    lambda_task_alignment = 0.3         # 任务对齐一致性权重

    # PCL扰动设置
    feature_dropout_rate = 0.1
    feature_noise_std = 0.01

    # ============ 评估设置 ============
    eval_batch_size = 64  # 评估时可以用更大的batch
    threshold = 0.5  # 多标签分类阈值

    # ============ 日志设置 ============
    log_dir = './logs'
    save_dir = './checkpoints'
    log_interval = 100  # 每多少步打印一次
    eval_interval = 500  # 每多少步评估一次
    save_interval = 1000  # 每多少步保存一次

    # ============ 其他 ============
    # 根据设备自动设置workers（提升到8以充分利用CPU预处理数据）
    num_workers = 8 if torch.cuda.is_available() else 0

    # 是否使用预处理缓存（强烈推荐，训练速度提升5-10倍）
    use_cached_data = True
    cache_dir = './dataset/cache'

    def __repr__(self):
        attrs = '\n'.join([f'{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        return f"Config(\n{attrs}\n)"
