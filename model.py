
"""
模型架构：结合K-LJP和PCL的法律判决预测模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Dict, Tuple


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer用于学习标签关系"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, num_labels, hidden_dim] 标签查询
            key_value: [batch, seq_len, hidden_dim] 事实描述特征
        """
        # Self-attention: 学习标签间关系
        q_prime, _ = self.self_attn(query, query, query)
        query = self.norm1(query + self.dropout(q_prime))

        # Cross-attention: 学习标签-事实关系
        q_double_prime, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm2(query + self.dropout(q_double_prime))

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm3(query + self.dropout(ffn_out))

        return query


class LegalJudgmentModel(nn.Module):
    """法律判决预测模型（基于K-LJP架构）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoder
        if config.encoder_type == 'bert':
            self.encoder = BertModel.from_pretrained(config.pretrained_model)
            self.hidden_dim = self.encoder.config.hidden_size
        else:
            raise NotImplementedError(f"Encoder type {config.encoder_type} not implemented")

        # 标签嵌入初始化（可选：从标签定义文本初始化）
        self.article_embeddings = nn.Embedding(config.num_articles, self.hidden_dim)
        self.charge_embeddings = nn.Embedding(config.num_charges, self.hidden_dim)

        # Transformer Decoder Layers（用于学习标签关系）
        self.article_decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.hidden_dim, config.num_attention_heads, config.decoder_dropout)
            for _ in range(config.num_decoder_layers)
        ])

        self.charge_decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.hidden_dim, config.num_attention_heads, config.decoder_dropout)
            for _ in range(config.num_decoder_layers)
        ])

        # Predictors
        self.article_predictor = nn.Linear(self.hidden_dim, 1)
        self.charge_predictor = nn.Linear(self.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.article_embeddings.weight)
        nn.init.xavier_uniform_(self.charge_embeddings.weight)
        nn.init.xavier_uniform_(self.article_predictor.weight)
        nn.init.xavier_uniform_(self.charge_predictor.weight)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码事实描述
        Returns:
            [batch, seq_len, hidden_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            return_features: 是否返回中间特征（用于扰动）
        Returns:
            {
                'article_logits': [batch, num_articles],
                'charge_logits': [batch, num_charges],
                'article_probs': [batch, num_articles],
                'charge_probs': [batch, num_charges],
                (可选) 'features': {...}  # 如果return_features=True
            }
        """
        batch_size = input_ids.size(0)

        # 编码事实描述
        fact_hidden = self.encode(input_ids, attention_mask)  # [batch, seq_len, hidden_dim]

        # 初始标签查询
        article_indices = torch.arange(self.config.num_articles, device=input_ids.device)
        charge_indices = torch.arange(self.config.num_charges, device=input_ids.device)

        article_queries = self.article_embeddings(article_indices).unsqueeze(0).expand(batch_size, -1, -1)
        charge_queries = self.charge_embeddings(charge_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # 通过Decoder学习标签-事实关系
        for layer in self.article_decoder_layers:
            article_queries = layer(article_queries, fact_hidden)

        for layer in self.charge_decoder_layers:
            charge_queries = layer(charge_queries, fact_hidden)

        # 预测
        article_logits = self.article_predictor(article_queries).squeeze(-1)  # [batch, num_articles]
        charge_logits = self.charge_predictor(charge_queries).squeeze(-1)     # [batch, num_charges]

        article_probs = torch.sigmoid(article_logits)
        charge_probs = torch.sigmoid(charge_logits)

        output = {
            'article_logits': article_logits,
            'charge_logits': charge_logits,
            'article_probs': article_probs,
            'charge_probs': charge_probs
        }

        if return_features:
            output['features'] = {
                'fact_hidden': fact_hidden,
                'article_queries': article_queries,
                'charge_queries': charge_queries
            }

        return output

    def forward_with_feature_perturbation(self, input_ids: torch.Tensor,
                                         attention_mask: torch.Tensor,
                                         dropout_rate: float = 0.1,
                                         noise_std: float = 0.01) -> Tuple[Dict, Dict]:
        """
        前向传播 + 特征扰动（用于PCL）
        Returns:
            (clean_output, perturbed_output)
        """
        batch_size = input_ids.size(0)

        # 编码事实描述
        fact_hidden = self.encode(input_ids, attention_mask)

        # 添加扰动
        fact_hidden_perturbed = F.dropout(fact_hidden, p=dropout_rate, training=True)
        noise = torch.randn_like(fact_hidden_perturbed) * noise_std
        fact_hidden_perturbed = fact_hidden_perturbed + noise

        # 原始预测
        article_indices = torch.arange(self.config.num_articles, device=input_ids.device)
        charge_indices = torch.arange(self.config.num_charges, device=input_ids.device)

        article_queries = self.article_embeddings(article_indices).unsqueeze(0).expand(batch_size, -1, -1)
        charge_queries = self.charge_embeddings(charge_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # Clean forward
        article_queries_clean = article_queries.clone()
        charge_queries_clean = charge_queries.clone()

        for layer in self.article_decoder_layers:
            article_queries_clean = layer(article_queries_clean, fact_hidden)

        for layer in self.charge_decoder_layers:
            charge_queries_clean = layer(charge_queries_clean, fact_hidden)

        article_logits_clean = self.article_predictor(article_queries_clean).squeeze(-1)
        charge_logits_clean = self.charge_predictor(charge_queries_clean).squeeze(-1)

        # Perturbed forward
        article_queries_perturbed = article_queries.clone()
        charge_queries_perturbed = charge_queries.clone()

        for layer in self.article_decoder_layers:
            article_queries_perturbed = layer(article_queries_perturbed, fact_hidden_perturbed)

        for layer in self.charge_decoder_layers:
            charge_queries_perturbed = layer(charge_queries_perturbed, fact_hidden_perturbed)

        article_logits_perturbed = self.article_predictor(article_queries_perturbed).squeeze(-1)
        charge_logits_perturbed = self.charge_predictor(charge_queries_perturbed).squeeze(-1)

        clean_output = {
            'article_logits': article_logits_clean,
            'charge_logits': charge_logits_clean,
            'article_probs': torch.sigmoid(article_logits_clean),
            'charge_probs': torch.sigmoid(charge_logits_clean)
        }

        perturbed_output = {
            'article_logits': article_logits_perturbed,
            'charge_logits': charge_logits_perturbed,
            'article_probs': torch.sigmoid(article_logits_perturbed),
            'charge_probs': torch.sigmoid(charge_logits_perturbed)
        }

        return clean_output, perturbed_output


if __name__ == "__main__":
    # 测试模型
    from config import Config

    config = Config()
    model = LegalJudgmentModel(config)

    # 模拟输入
    batch_size = 4
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)

    # 前向传播
    output = model(input_ids, attention_mask)

    print("Article probs shape:", output['article_probs'].shape)
    print("Charge probs shape:", output['charge_probs'].shape)

    # 测试特征扰动
    clean, perturbed = model.forward_with_feature_perturbation(input_ids, attention_mask)
    print("Feature perturbation test passed!")
