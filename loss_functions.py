"""
创新点实现：
1. 动态加权
2. 不对称更新
3. 置信度门控
4. 对齐损失计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from data_loader import MappingDictionary


class AlignmentMetrics:
    """对齐度计算"""

    @staticmethod
    def compute_sample_alignment(article_probs: torch.Tensor,
                                 charge_probs: torch.Tensor,
                                 mapping_dict: MappingDictionary,
                                 threshold: float = 0.5) -> torch.Tensor:
        """
        计算样本级别的对齐度
        Args:
            article_probs: [batch, num_articles]
            charge_probs: [batch, num_charges]
            mapping_dict: 映射字典
            threshold: 预测阈值
        Returns:
            [batch] 每个样本的对齐度分数
        """
        batch_size = article_probs.size(0)
        device = article_probs.device

        # 二值化预测
        articles_pred = (article_probs > threshold).float()
        charges_pred = (charge_probs > threshold).float()

        # 映射
        mapped_charges = mapping_dict.articles_to_charges_tensor(articles_pred, device)
        mapped_articles = mapping_dict.charges_to_articles_tensor(charges_pred, device)

        # 对齐度 = (映射一致的标签数) / (总预测标签数)
        align_score = []
        for i in range(batch_size):
            article_align = torch.sum(articles_pred[i] * (mapped_articles[i] > threshold).float())
            charge_align = torch.sum(charges_pred[i] * (mapped_charges[i] > threshold).float())
            total_pred = torch.sum(articles_pred[i]) + torch.sum(charges_pred[i])

            if total_pred > 0:
                score = (article_align + charge_align) / total_pred
            else:
                score = torch.tensor(0.0, device=device)

            align_score.append(score)

        return torch.stack(align_score)


class DynamicWeighting:
    """创新点1：动态加权机制"""

    def __init__(self, base_lambda: float = 1.0, sensitivity: float = 2.0):
        self.base_lambda = base_lambda
        self.sensitivity = sensitivity

    def compute_lambda(self, align_score: torch.Tensor) -> torch.Tensor:
        """
        根据对齐度动态计算权重
        对齐度高 → λ降低（减少更新强度）
        对齐度低 → λ增加（加大跨任务修复）

        Args:
            align_score: [batch] 对齐度分数
        Returns:
            [batch] 动态权重
        """
        # 使用sigmoid函数：对齐度越低，λ越大
        lambda_dynamic = self.base_lambda * torch.sigmoid(
            self.sensitivity * (0.5 - align_score)
        )
        return lambda_dynamic


class ConfidenceGating:
    """创新点3：置信度门控机制"""

    def __init__(self, conf_high: float = 0.7, conf_low: float = 0.4,
                 consistency_threshold: float = 0.3):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.consistency_threshold = consistency_threshold

    def compute_confidence(self, probs: torch.Tensor, method: str = 'max_prob') -> torch.Tensor:
        """
        计算预测置信度
        Args:
            probs: [batch, num_labels]
            method: 'max_prob' 或 'entropy'
        Returns:
            [batch] 置信度分数
        """
        if method == 'max_prob':
            # 方法1：最大概率的平均值
            max_probs, _ = torch.max(probs, dim=1)
            confidence = max_probs

        elif method == 'entropy':
            # 方法2：熵的倒数
            epsilon = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + epsilon) +
                               (1 - probs) * torch.log(1 - probs + epsilon), dim=1)
            confidence = 1.0 / (1.0 + entropy)

        else:
            raise ValueError(f"Unknown method: {method}")

        return confidence

    def should_update(self, article_probs: torch.Tensor,
                     charge_probs: torch.Tensor,
                     align_score: torch.Tensor = None) -> torch.Tensor:
        """
        判断是否应该对样本执行TTA更新
        Args:
            article_probs: [batch, num_articles]
            charge_probs: [batch, num_charges]
            align_score: [batch] 对齐度（可选）
        Returns:
            [batch] bool tensor，True表示应该更新
        """
        # 计算综合置信度
        conf_articles = self.compute_confidence(article_probs)
        conf_charges = self.compute_confidence(charge_probs)
        overall_conf = (conf_articles + conf_charges) / 2

        # 规则1：保护高置信度样本
        high_conf_mask = overall_conf > self.conf_high

        # 规则2：只更新低置信度样本
        low_conf_mask = overall_conf < self.conf_low

        # 规则3：一致性门控（如果提供对齐度）
        if align_score is not None:
            # 危险信号：高置信度 + 低对齐度 = 可能的系统性错误
            dangerous_mask = (overall_conf > self.conf_high) & (align_score < self.consistency_threshold)
            should_update_mask = low_conf_mask & (~dangerous_mask)
        else:
            should_update_mask = low_conf_mask

        # 高置信度样本不更新
        should_update_mask = should_update_mask & (~high_conf_mask)

        return should_update_mask


class LossCalculator:
    """损失计算器"""

    def __init__(self, config, mapping_dict: MappingDictionary):
        self.config = config
        self.mapping_dict = mapping_dict

        # 创新点1：动态加权
        if config.use_dynamic_weighting:
            self.dynamic_weighting = DynamicWeighting(
                base_lambda=config.base_lambda,
                sensitivity=config.sensitivity
            )
        else:
            self.dynamic_weighting = None

    def compute_cross_entropy_loss(self, pred_probs: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        """
        多标签二元交叉熵损失
        Args:
            pred_probs: [batch, num_labels]
            target: [batch, num_labels]
        """
        epsilon = 1e-8
        loss = -torch.mean(
            target * torch.log(pred_probs + epsilon) +
            (1 - target) * torch.log(1 - pred_probs + epsilon)
        )
        return loss

    def compute_asymmetric_alignment_loss(self, article_probs: torch.Tensor,
                                         charge_probs: torch.Tensor) -> torch.Tensor:
        """
        创新点2：不对称对齐损失
        让罪名更多地纠正法条，而不是对称约束

        Args:
            article_probs: [batch, num_articles]
            charge_probs: [batch, num_charges]
        Returns:
            对齐损失
        """
        device = article_probs.device

        # 罪名 → 法条：强约束（让法条逼近罪名映射的分布）
        # 使用 detach() 防止罪名预测被拉偏
        mapped_articles_from_charges = self.mapping_dict.charges_to_articles_tensor(
            charge_probs.detach(), device
        )

        L_charge_to_article = F.kl_div(
            torch.log(article_probs + 1e-8),
            mapped_articles_from_charges,
            reduction='batchmean'
        )

        # 法条 → 罪名：弱约束
        mapped_charges_from_articles = self.mapping_dict.articles_to_charges_tensor(
            article_probs, device
        )

        L_article_to_charge = F.kl_div(
            torch.log(charge_probs + 1e-8),
            mapped_charges_from_articles,
            reduction='batchmean'
        )

        # 不对称权重
        alpha = self.config.alpha  # 罪名引导法条的权重（更高）
        beta = self.config.beta    # 法条引导罪名的权重

        return alpha * L_charge_to_article + beta * L_article_to_charge

    def compute_symmetric_alignment_loss(self, article_probs: torch.Tensor,
                                        charge_probs: torch.Tensor) -> torch.Tensor:
        """
        对称对齐损失（K-LJP原始方法）
        """
        device = article_probs.device

        # 法条 → 罪名映射
        mapped_charges = self.mapping_dict.articles_to_charges_tensor(article_probs, device)
        L_a2c = F.kl_div(
            torch.log(mapped_charges + 1e-8),
            charge_probs,
            reduction='batchmean'
        )

        # 罪名 → 法条映射
        mapped_articles = self.mapping_dict.charges_to_articles_tensor(charge_probs, device)
        L_c2a = F.kl_div(
            torch.log(mapped_articles + 1e-8),
            article_probs,
            reduction='batchmean'
        )

        return (L_a2c + L_c2a) / 2

    def compute_pcl_loss(self, clean_output: Dict, perturbed_output: Dict) -> torch.Tensor:
        """
        PCL损失：特征扰动一致性
        Args:
            clean_output: 原始输出
            perturbed_output: 扰动后输出
        """
        # 法条预测一致性
        L_pcl_articles = F.kl_div(
            torch.log(perturbed_output['article_probs'] + 1e-8),
            clean_output['article_probs'].detach(),
            reduction='batchmean'
        )

        # 罪名预测一致性
        L_pcl_charges = F.kl_div(
            torch.log(perturbed_output['charge_probs'] + 1e-8),
            clean_output['charge_probs'].detach(),
            reduction='batchmean'
        )

        return L_pcl_articles + L_pcl_charges

    def compute_total_loss(self, model_output: Dict, labels: Dict,
                          with_pcl: bool = False,
                          perturbed_output: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失（集成所有创新点）
        Args:
            model_output: 模型输出
            labels: 标签 {'articles': ..., 'charges': ...}
            with_pcl: 是否使用PCL损失
            perturbed_output: 扰动输出（如果with_pcl=True）
        Returns:
            (total_loss, loss_dict)
        """
        article_probs = model_output['article_probs']
        charge_probs = model_output['charge_probs']

        # 1. 任务内损失（交叉熵）
        L_ce_articles = self.compute_cross_entropy_loss(article_probs, labels['articles'])
        L_ce_charges = self.compute_cross_entropy_loss(charge_probs, labels['charges'])
        L_ce = L_ce_articles + L_ce_charges

        # 2. 任务间对齐损失
        if self.config.use_asymmetric_alignment:
            L_align = self.compute_asymmetric_alignment_loss(article_probs, charge_probs)
        else:
            L_align = self.compute_symmetric_alignment_loss(article_probs, charge_probs)

        # 3. 动态加权（如果启用）
        if self.dynamic_weighting is not None:
            align_scores = AlignmentMetrics.compute_sample_alignment(
                article_probs, charge_probs, self.mapping_dict
            )
            lambda_dynamic = self.dynamic_weighting.compute_lambda(align_scores).mean()
            L_align = lambda_dynamic * L_align

        # 4. PCL损失（如果启用）
        if with_pcl and perturbed_output is not None:
            L_pcl = self.compute_pcl_loss(model_output, perturbed_output)
            total_loss = L_ce + L_align + self.config.lambda_feature_perturbation * L_pcl
            loss_dict = {
                'loss': total_loss.item(),
                'ce_loss': L_ce.item(),
                'align_loss': L_align.item(),
                'pcl_loss': L_pcl.item()
            }
        else:
            total_loss = L_ce + L_align
            loss_dict = {
                'loss': total_loss.item(),
                'ce_loss': L_ce.item(),
                'align_loss': L_align.item()
            }

        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append('.')
    from config import Config

    config = Config()
    config.use_dynamic_weighting = True
    config.use_asymmetric_alignment = True

    # 创建虚拟映射字典
    class DummyMapping:
        def articles_to_charges_tensor(self, probs, device):
            return torch.sigmoid(torch.randn_like(probs))

        def charges_to_articles_tensor(self, probs, device):
            return torch.sigmoid(torch.randn_like(probs))

    mapping_dict = DummyMapping()
    loss_calculator = LossCalculator(config, mapping_dict)

    # 测试
    batch_size = 4
    article_probs = torch.sigmoid(torch.randn(batch_size, 121))
    charge_probs = torch.sigmoid(torch.randn(batch_size, 150))

    model_output = {
        'article_probs': article_probs,
        'charge_probs': charge_probs
    }

    labels = {
        'articles': torch.randint(0, 2, (batch_size, 121)).float(),
        'charges': torch.randint(0, 2, (batch_size, 150)).float()
    }

    loss, loss_dict = loss_calculator.compute_total_loss(model_output, labels)
    print("Loss test passed!")
    print("Loss dict:", loss_dict)
