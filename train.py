"""
主训练脚本：训练法律判决预测模型
"""
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
import json
from pathlib import Path

from config import Config
from model import LegalJudgmentModel
from data_loader import get_dataloader, MappingDictionary
from loss_functions import LossCalculator
from evaluator import Evaluator
from tta_trainer import TTATrainer


class Trainer:
    """主训练器"""

    def __init__(self, config: Config):
        self.config = config

        # 设置随机种子
        self._set_seed(config.seed)

        # 创建目录
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)

        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

        # 加载映射字典
        self.mapping_dict = MappingDictionary(
            config.a2c_dict_file,
            config.c2a_dict_file
        )

        # 初始化模型
        self.model = LegalJudgmentModel(config).to(config.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        # 损失计算器
        self.loss_calculator = LossCalculator(config, self.mapping_dict)

        # 评估器
        self.evaluator = Evaluator(threshold=config.threshold)

        # TensorBoard
        self.writer = SummaryWriter(config.log_dir)

        # 训练状态
        self.global_step = 0
        self.best_metric = 0.0

    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)

    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        loss_stats = {'ce_loss': 0.0, 'align_loss': 0.0, 'pcl_loss': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            article_labels = batch['articles'].to(self.config.device)
            charge_labels = batch['charges'].to(self.config.device)

            labels = {
                'articles': article_labels,
                'charges': charge_labels
            }

            # 前向传播
            if self.config.use_triple_consistency:
                # 使用特征扰动
                clean_output, perturbed_output = self.model.forward_with_feature_perturbation(
                    input_ids, attention_mask,
                    dropout_rate=self.config.feature_dropout_rate,
                    noise_std=self.config.feature_noise_std
                )
                loss, loss_dict = self.loss_calculator.compute_total_loss(
                    clean_output, labels, with_pcl=True, perturbed_output=perturbed_output
                )
            else:
                # 标准训练
                output = self.model(input_ids, attention_mask)
                loss, loss_dict = self.loss_calculator.compute_total_loss(
                    output, labels, with_pcl=False
                )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # 统计
            total_loss += loss.item()
            for key in loss_dict:
                if key in loss_stats:
                    loss_stats[key] += loss_dict[key]

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # 记录到TensorBoard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                self.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], self.global_step)

            self.global_step += 1

        # 平均损失
        avg_loss = total_loss / len(train_loader)
        for key in loss_stats:
            loss_stats[key] /= len(train_loader)

        return avg_loss, loss_stats

    def evaluate(self, data_loader, desc: str = "Evaluating"):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)

                output = self.model(input_ids, attention_mask)

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

        # 计算指标
        metrics = self.evaluator.compute_metrics(all_predictions, all_labels)
        align_ratio = self.evaluator.compute_alignment_ratio(
            all_predictions, all_labels, self.mapping_dict
        )
        metrics['alignment_ratio'] = align_ratio

        return metrics, all_predictions, all_labels

    def train(self, train_loader, dev_loader):
        """完整训练流程"""
        print("\n" + "="*60)
        print("🚀 Starting Training")
        print("="*60)
        print(f"Device: {self.config.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Dynamic weighting: {self.config.use_dynamic_weighting}")
        print(f"Asymmetric alignment: {self.config.use_asymmetric_alignment}")
        print(f"Triple consistency: {self.config.use_triple_consistency}")
        print("="*60 + "\n")

        # 优化器和调度器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # 训练循环
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*60}")

            # 训练
            train_loss, loss_stats = self.train_epoch(train_loader, optimizer, scheduler, epoch)

            print(f"\n📊 Epoch {epoch} Training Summary:")
            print(f"  Average Loss: {train_loss:.4f}")
            for key, value in loss_stats.items():
                if value > 0:
                    print(f"  {key}: {value:.4f}")

            # 评估
            print(f"\n🔍 Evaluating on development set...")
            dev_metrics, _, _ = self.evaluate(dev_loader, desc="Dev Eval")

            # 打印评估结果
            self.evaluator.print_metrics(dev_metrics, f"Epoch {epoch} Dev")

            # 记录到TensorBoard
            for key, value in dev_metrics.items():
                self.writer.add_scalar(f'dev/{key}', value, epoch)

            # 保存最佳模型
            current_metric = dev_metrics['avg_ma_f1']
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model saved! Avg Macro-F1: {current_metric:.4f}")

            # 定期保存
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "="*60)
        print("✨ Training completed!")
        print(f"Best Avg Macro-F1: {self.best_metric:.4f}")
        print("="*60 + "\n")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def main():
    """主函数"""
    config = Config()

    # 创建数据加载器
    print("📂 Loading data...")
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

    train_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.train_file),
        tokenizer,
        shuffle=True
    )

    dev_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.dev_file),
        tokenizer,
        shuffle=False
    )

    test_loader = get_dataloader(
        config,
        os.path.join(config.data_dir, config.test_file),
        tokenizer,
        shuffle=False
    )

    # 创建训练器
    trainer = Trainer(config)

    # 训练
    trainer.train(train_loader, dev_loader)

    # 加载最佳模型
    best_model_path = os.path.join(config.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    # 最终测试
    print("\n🎯 Final evaluation on test set...")
    test_metrics, test_preds, test_labels = trainer.evaluate(test_loader, desc="Test Eval")
    trainer.evaluator.print_metrics(test_metrics, "Final Test")

    # 保存测试结果
    results = {
        'config': str(config),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }

    results_path = os.path.join(config.save_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Results saved to {results_path}")


if __name__ == "__main__":
    main()
