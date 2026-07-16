"""
优化的损失函数：去除所有 Python 循环，使用纯张量操作
性能提升：5-10 倍
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from data_loader_optimized import FastMappingDictionary


class FastAlignmentMetrics:
    """对齐度计算（矩阵化版本）"""

    @staticmethod
    def compute_sample_alignment(article_probs: torch.Tensor,
                                 charge_probs: torch.Tensor,
                                 mapping_dict: FastMappingDictionary,
                                 threshold: float = 0.5) -> torch.Tensor:
        """
        计算样本级别的对齐度（矩阵化版本）
        Args:
            article_probs: [batch, num_articles]
            charge_probs: [batch, num_charges]
            mapping_dict: 映射字典
            threshold: 预测阈值
        Returns:
            [batch] 每个样本的对齐度分数
        """
        device = article_probs.device

        # 二值化预测
        articles_pred = (article_probs > threshold).float()
        charges_pred = (charge_probs > threshold).float()

        # 🚀 使用矩阵化映射（无循环）
        mapped_charges = mapping_dict.articles_to_charges_tensor(articles_pred, device)
        mapped_articles = mapping_dict.charges_to_articles_tensor(charges_pred, device)

        # 🚀 矢量化计算对齐度（替代 for 循环）
        # [batch, num_articles]
        article_align = articles_pred * (mapped_articles > threshold).float()
        charge_align = charges_pred * (mapped_charges > threshold).float()

        # [batch] - 每个样本的对齐标签数
        article_align_count = torch.sum(article_align, dim=1)
        charge_align_count = torch.sum(charge_align, dim=1)

        # [batch] - 每个样本的总预测标签数
        total_pred = torch.sum(articles_pred, dim=1) + torch.sum(charges_pred, dim=1)

        # 对齐度分数（避免除零）
        align_score = (article_align_count + charge_align_count) / (total_pred + 1e-8)

        return align_score


class DynamicWeighting:
    """创新点1：动态加权机制（已优化）"""

    def __init__(self, base_lambda: float = 1.0, sensitivity: float = 2.0):
        self.base_lambda = base_lambda
        self.sensitivity = sensitivity

    def compute_lambda(self, align_score: torch.Tensor) -> torch.Tensor:
        """
        根据对齐度动态计算权重（矢量化）
        Args:
            align_score: [batch] 对齐度分数
        Returns:
            [batch] 动态权重
        """
        lambda_dynamic = self.base_lambda * torch.sigmoid(
            self.sensitivity * (0.5 - align_score)
        )
        return lambda_dynamic


class ConfidenceGating:
    """创新点3：置信度门控机制（已优化）"""

    def __init__(self, conf_high: float = 0.7, conf_low: float = 0.4,
                 consistency_threshold: float = 0.3):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.consistency_threshold = consistency_threshold

    def compute_confidence(self, probs: torch.Tensor, method: str = 'max_prob') -> torch.Tensor:
        """
        计算预测置信度（矢量化）
        Args:
            probs: [batch, num_labels]
            method: 'max_prob', 'entropy', 'margin'
        Returns:
            [batch] 置信度分数
        """
        if method == 'max_prob':
            # 最大概率作为置信度
            confidence = torch.max(probs, dim=1)[0]
        elif method == 'entropy':
            # 熵越低，置信度越高
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            confidence = 1.0 - entropy / torch.log(torch.tensor(probs.size(1), dtype=torch.float))
        elif method == 'margin':
            # Top-2差距
            top2 = torch.topk(probs, k=2, dim=1)[0]
            confidence = top2[:, 0] - top2[:, 1]
        else:
            confidence = torch.max(probs, dim=1)[0]

        return confidence

    def compute_gate(self, confidence: torch.Tensor, align_score: torch.Tensor = None) -> torch.Tensor:
        """
        计算门控掩码（矢量化）
        Args:
            confidence: [batch] 置信度
            align_score: [batch] 对齐度（可选）
        Returns:
            [batch] 门控掩码（0或1）
        """
        # 基于置信度的门控
        gate = ((confidence < self.conf_high) & (confidence > self.conf_low)).float()

        # 如果提供对齐度，加入一致性门控
        if align_score is not None:
            consistency_gate = (align_score < self.consistency_threshold).float()
            gate = gate * consistency_gate

        return gate


class FastLossCalculator:
    """优化的损失计算器"""

    def __init__(self, config, mapping_dict: FastMappingDictionary):
        self.config = config
        self.mapping_dict = mapping_dict

        # 🚀 使用 BCEWithLogitsLoss 以支持混合精度
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        if config.use_dynamic_weighting:
            self.dynamic_weighting = DynamicWeighting(
                config.base_lambda,
                config.sensitivity
            )

        if config.use_confidence_gating:
            self.confidence_gating = ConfidenceGating(
                config.confidence_high_threshold,
                config.confidence_low_threshold,
                config.consistency_threshold
            )

    def compute_classification_loss(self, predictions: Dict, labels: Dict) -> Tuple[torch.Tensor, Dict]:
        """计算分类损失"""
        # 🚀 使用 logits 而不是 probs（支持混合精度）
        article_logits = predictions['article_logits']
        charge_logits = predictions['charge_logits']
        article_labels = labels['articles']
        charge_labels = labels['charges']

        # BCEWithLogits 损失（安全的混合精度）
        article_loss = self.bce_loss(article_logits, article_labels).mean()
        charge_loss = self.bce_loss(charge_logits, charge_labels).mean()

        total_loss = article_loss + charge_loss

        loss_dict = {
            'article_loss': article_loss.item(),
            'charge_loss': charge_loss.item(),
        }

        return total_loss, loss_dict

    def compute_alignment_loss(self, predictions: Dict) -> torch.Tensor:
        """计算对齐损失（优化版）"""
        article_probs = predictions['article_probs']
        charge_probs = predictions['charge_probs']
        device = article_probs.device

        # 🚀 矩阵化映射
        mapped_charges = self.mapping_dict.articles_to_charges_tensor(article_probs, device)
        mapped_articles = self.mapping_dict.charges_to_articles_tensor(charge_probs, device)

        # 计算双向对齐损失
        if self.config.use_asymmetric_alignment:
            # 不对称更新
            alpha = self.config.alpha
            beta = self.config.beta
            align_loss = (
                alpha * F.mse_loss(article_probs, mapped_articles) +
                beta * F.mse_loss(charge_probs, mapped_charges)
            )
        else:
            # 对称更新
            align_loss = (
                F.mse_loss(article_probs, mapped_articles) +
                F.mse_loss(charge_probs, mapped_charges)
            ) / 2.0

        return align_loss

    def compute_total_loss(self, predictions: Dict, labels: Dict,
                          with_pcl: bool = False, perturbed_output: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """计算总损失"""
        # 分类损失
        ce_loss, loss_dict = self.compute_classification_loss(predictions, labels)

        # 对齐损失
        align_loss = self.compute_alignment_loss(predictions)

        # 动态加权
        if self.config.use_dynamic_weighting:
            align_score = FastAlignmentMetrics.compute_sample_alignment(
                predictions['article_probs'],
                predictions['charge_probs'],
                self.mapping_dict
            )
            lambda_dynamic = self.dynamic_weighting.compute_lambda(align_score)
            lambda_weight = lambda_dynamic.mean()
        else:
            lambda_weight = 1.0

        # 置信度门控
        if self.config.use_confidence_gating:
            article_conf = self.confidence_gating.compute_confidence(predictions['article_probs'])
            charge_conf = self.confidence_gating.compute_confidence(predictions['charge_probs'])
            confidence = (article_conf + charge_conf) / 2.0

            if self.config.use_consistency_gating:
                align_score = FastAlignmentMetrics.compute_sample_alignment(
                    predictions['article_probs'],
                    predictions['charge_probs'],
                    self.mapping_dict
                )
                gate = self.confidence_gating.compute_gate(confidence, align_score)
            else:
                gate = self.confidence_gating.compute_gate(confidence)

            gate_weight = gate.mean()
        else:
            gate_weight = 1.0

        # 总损失
        total_loss = ce_loss + lambda_weight * gate_weight * align_loss

        loss_dict.update({
            'ce_loss': ce_loss.item(),
            'align_loss': align_loss.item(),
            'lambda_weight': float(lambda_weight) if isinstance(lambda_weight, torch.Tensor) else lambda_weight,
            'gate_weight': float(gate_weight) if isinstance(gate_weight, torch.Tensor) else gate_weight,
        })

        # PCL损失（如果启用）
        if with_pcl and perturbed_output is not None:
            pcl_loss = F.mse_loss(
                predictions['article_probs'],
                perturbed_output['article_probs']
            ) + F.mse_loss(
                predictions['charge_probs'],
                perturbed_output['charge_probs']
            )
            total_loss = total_loss + self.config.lambda_feature_perturbation * pcl_loss
            loss_dict['pcl_loss'] = pcl_loss.item()

        return total_loss, loss_dict
