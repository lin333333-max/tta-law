"""
TTA (Test-Time Adaptation) 训练器
实现创新点的测试时适应策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from tqdm import tqdm
import copy

from model import LegalJudgmentModel
from loss_functions import LossCalculator, ConfidenceGating, AlignmentMetrics
from data_loader import MappingDictionary


class TTATrainer:
    """测试时适应训练器"""

    def __init__(self, model: LegalJudgmentModel, config, mapping_dict: MappingDictionary):
        self.model = model
        self.config = config
        self.mapping_dict = mapping_dict

        # 损失计算器
        self.loss_calculator = LossCalculator(config, mapping_dict)

        # 置信度门控（创新点3）
        if config.use_confidence_gating:
            self.confidence_gating = ConfidenceGating(
                conf_high=config.confidence_high_threshold,
                conf_low=config.confidence_low_threshold,
                consistency_threshold=config.consistency_threshold
            )
        else:
            self.confidence_gating = None

        # 统计信息
        self.stats = {
            'total_samples': 0,
            'updated_samples': 0,
            'skipped_high_conf': 0,
            'skipped_dangerous': 0
        }

    def setup_optimizer(self):
        """
        设置TTA优化器
        参考PCL论文：TTA阶段只更新LayerNorm参数（而非全参数），
        既能保留大部分预训练知识，又能大幅降低显存开销（无需为全部参数维护Adam动量）
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            is_layernorm = 'norm' in name.lower()  # 匹配 BERT 的 LayerNorm.* 和 Decoder 的 norm1/2/3
            is_charge_frozen = self.config.freeze_charge_in_tta and 'charge' in name.lower()

            if self.config.tta_update_layernorm_only:
                param.requires_grad = is_layernorm and not is_charge_frozen
            else:
                param.requires_grad = not is_charge_frozen

            if param.requires_grad:
                params_to_update.append(param)

        optimizer = optim.Adam(params_to_update, lr=self.config.tta_learning_rate)
        return optimizer

    def tta_update_sample(self, batch: Dict, num_steps: int = 5) -> Dict:
        """
        对单个batch执行TTA更新

        Args:
            batch: 包含输入数据的字典
            num_steps: TTA更新步数
        Returns:
            更新后的预测结果
        """
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)

        self.model.train()  # 启用dropout等

        # 设置优化器（只更新LayerNorm等少量参数，见setup_optimizer）
        optimizer = self.setup_optimizer()
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]

        # 初始预测（用于判断是否需要更新）
        with torch.no_grad():
            initial_output = self.model(input_ids, attention_mask)

        # 创新点3：置信度门控判断是否更新
        if self.confidence_gating is not None:
            align_score = None
            if self.config.use_consistency_gating:
                align_score = AlignmentMetrics.compute_sample_alignment(
                    initial_output['article_probs'],
                    initial_output['charge_probs'],
                    self.mapping_dict
                )

            should_update_mask = self.confidence_gating.should_update(
                initial_output['article_probs'],
                initial_output['charge_probs'],
                align_score
            )

            # 统计
            self.stats['total_samples'] += len(should_update_mask)
            self.stats['updated_samples'] += should_update_mask.sum().item()
            self.stats['skipped_high_conf'] += (~should_update_mask).sum().item()

            # 如果所有样本都不需要更新，直接返回
            if not should_update_mask.any():
                return initial_output
        else:
            should_update_mask = torch.ones(input_ids.size(0), dtype=torch.bool)

        # 执行TTA更新
        for step in range(num_steps):
            optimizer.zero_grad()

            if self.config.use_triple_consistency:
                # 使用特征扰动（PCL）
                clean_output, perturbed_output = self.model.forward_with_feature_perturbation(
                    input_ids, attention_mask,
                    dropout_rate=self.config.feature_dropout_rate,
                    noise_std=self.config.feature_noise_std
                )

                # 计算损失（只对需要更新的样本）
                # 这里简化处理：对整个batch计算损失，可以优化为只计算需要更新的样本
                loss, loss_dict = self.loss_calculator.compute_total_loss(
                    clean_output,
                    labels=None,  # TTA无标签
                    with_pcl=True,
                    perturbed_output=perturbed_output
                )
            else:
                # 标准前向传播
                output = self.model(input_ids, attention_mask)

                # 只使用对齐损失（无监督）
                if self.config.use_asymmetric_alignment:
                    loss = self.loss_calculator.compute_asymmetric_alignment_loss(
                        output['article_probs'],
                        output['charge_probs']
                    )
                else:
                    loss = self.loss_calculator.compute_symmetric_alignment_loss(
                        output['article_probs'],
                        output['charge_probs']
                    )

            # 反向传播和更新
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, self.config.max_grad_norm)
            optimizer.step()

        # 释放优化器（Adam为每个参数维护动量状态，及时释放避免跨batch显存累积）
        del optimizer

        # 最终预测
        self.model.eval()
        with torch.no_grad():
            final_output = self.model(input_ids, attention_mask)

        return final_output

    def test_with_tta(self, test_loader, save_model_per_batch: bool = False):
        """
        在测试集上执行TTA

        Args:
            test_loader: 测试数据加载器
            save_model_per_batch: 是否每个batch后保存模型状态
        Returns:
            predictions: 预测结果列表
            labels: 真实标签列表
        """
        # 只备份会被TTA更新的参数（LayerNorm等），且放在CPU上，避免占用显存、
        # 避免每个batch都deepcopy整个模型state_dict（BERT全参数deepcopy 508次是OOM的主因）
        updatable_names = set()
        for name, param in self.model.named_parameters():
            is_layernorm = 'norm' in name.lower()
            is_charge_frozen = self.config.freeze_charge_in_tta and 'charge' in name.lower()
            if self.config.tta_update_layernorm_only:
                if is_layernorm and not is_charge_frozen:
                    updatable_names.add(name)
            elif not is_charge_frozen:
                updatable_names.add(name)

        original_state_cpu = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if name in updatable_names
        }

        all_predictions = []
        all_labels = []

        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="TTA Testing")):
            # 恢复被更新过的参数（如果不保存状态），而非整个模型state_dict
            if not save_model_per_batch:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in original_state_cpu:
                            param.copy_(original_state_cpu[name].to(param.device))

            # 执行TTA更新
            output = self.tta_update_sample(batch, num_steps=self.config.tta_steps)

            # 收集预测结果
            predictions = {
                'articles': output['article_probs'].cpu(),
                'charges': output['charge_probs'].cpu()
            }

            labels = {
                'articles': batch['articles'],
                'charges': batch['charges']
            }

            all_predictions.append(predictions)
            all_labels.append(labels)

        # 恢复原始模型状态
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_state_cpu:
                    param.copy_(original_state_cpu[name].to(param.device))

        return all_predictions, all_labels

    def print_stats(self):
        """打印统计信息"""
        if self.stats['total_samples'] > 0:
            update_rate = self.stats['updated_samples'] / self.stats['total_samples'] * 100
            print(f"\n=== TTA Statistics ===")
            print(f"Total samples: {self.stats['total_samples']}")
            print(f"Updated samples: {self.stats['updated_samples']} ({update_rate:.2f}%)")
            print(f"Skipped (high confidence): {self.stats['skipped_high_conf']}")
            print(f"=====================\n")


class OnlineTTATrainer(TTATrainer):
    """在线TTA训练器（模型状态持续更新，不重置）"""

    def test_with_tta(self, test_loader):
        """
        在线TTA：模型状态在整个测试集上累积更新
        """
        all_predictions = []
        all_labels = []

        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Online TTA Testing")):
            # 执行TTA更新（模型状态不重置）
            output = self.tta_update_sample(batch, num_steps=self.config.tta_steps)

            # 收集预测结果
            predictions = {
                'articles': output['article_probs'].cpu(),
                'charges': output['charge_probs'].cpu()
            }

            labels = {
                'articles': batch['articles'],
                'charges': batch['charges']
            }

            all_predictions.append(predictions)
            all_labels.append(labels)

            # 定期打印统计
            if (batch_idx + 1) % 100 == 0:
                print(f"\nBatch {batch_idx + 1}/{len(test_loader)}")
                self.print_stats()

        return all_predictions, all_labels


class AdaptiveTTATrainer(TTATrainer):
    """自适应TTA训练器（根据样本难度动态调整更新步数）"""

    def tta_update_sample(self, batch: Dict, num_steps: int = 5) -> Dict:
        """
        自适应TTA：根据置信度动态调整更新步数

        低置信度样本 -> 更多更新步数
        高置信度样本 -> 更少更新步数
        """
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)

        # 初始预测
        with torch.no_grad():
            initial_output = self.model(input_ids, attention_mask)

        # 根据置信度调整步数
        if self.confidence_gating is not None:
            conf_articles = self.confidence_gating.compute_confidence(
                initial_output['article_probs']
            )
            conf_charges = self.confidence_gating.compute_confidence(
                initial_output['charge_probs']
            )
            overall_conf = (conf_articles + conf_charges) / 2

            # 动态步数：置信度低 -> 更多步数
            adaptive_steps = []
            for conf in overall_conf:
                if conf < 0.3:
                    steps = num_steps * 2  # 非常不确定，加倍更新
                elif conf < 0.5:
                    steps = num_steps
                elif conf < 0.7:
                    steps = max(1, num_steps // 2)  # 较确定，减少更新
                else:
                    steps = 0  # 非常确定，不更新

                adaptive_steps.append(int(steps))

            # 使用最大步数（批次内）
            actual_steps = max(adaptive_steps) if adaptive_steps else num_steps
        else:
            actual_steps = num_steps

        # 调用父类方法执行更新
        if actual_steps > 0:
            return super().tta_update_sample(batch, num_steps=actual_steps)
        else:
            return initial_output


if __name__ == "__main__":
    # 测试代码
    from config import Config
    from model import LegalJudgmentModel

    config = Config()
    model = LegalJudgmentModel(config).to(config.device)

    # 创建虚拟映射字典
    class DummyMapping:
        def articles_to_charges_tensor(self, probs, device):
            return torch.sigmoid(torch.randn(probs.size(0), 150, device=device))

        def charges_to_articles_tensor(self, probs, device):
            return torch.sigmoid(torch.randn(probs.size(0), 121, device=device))

    mapping_dict = DummyMapping()

    # 创建TTA训练器
    tta_trainer = TTATrainer(model, config, mapping_dict)

    # 虚拟batch
    batch = {
        'input_ids': torch.randint(0, 1000, (4, 128)).to(config.device),
        'attention_mask': torch.ones(4, 128).to(config.device)
    }

    # 测试TTA更新
    output = tta_trainer.tta_update_sample(batch, num_steps=3)
    print("TTA update test passed!")
    print("Output keys:", output.keys())

    tta_trainer.print_stats()
