"""
使用自动混合精度(AMP)训练 - 提速2-3倍，精度损失<0.5%
"""
import os
import sys

# 在最开始就设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # 混合精度
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
import json

from config import Config
from model import LegalJudgmentModel
from fast_data_loader import get_fast_dataloader
from data_loader import MappingDictionary
from loss_functions import LossCalculator
from evaluator import Evaluator


class AMPTrainer:
    """使用混合精度的训练器"""

    def __init__(self, config: Config):
        self.config = config
        self._set_seed(config.seed)

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.mapping_dict = MappingDictionary(config.a2c_dict_file, config.c2a_dict_file)
        self.model = LegalJudgmentModel(config).to(config.device)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        self.loss_calculator = LossCalculator(config, self.mapping_dict)
        self.evaluator = Evaluator(threshold=config.threshold)
        self.writer = SummaryWriter(config.log_dir)

        # 🚀 混合精度训练的关键：GradScaler
        self.scaler = GradScaler()

        self.global_step = 0
        self.best_metric = 0.0

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)

    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int):
        """训练一个epoch（使用混合精度）"""
        self.model.train()
        total_loss = 0.0
        loss_stats = {'ce_loss': 0.0, 'align_loss': 0.0, 'pcl_loss': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [AMP]")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            article_labels = batch['articles'].to(self.config.device)
            charge_labels = batch['charges'].to(self.config.device)

            labels = {
                'articles': article_labels,
                'charges': charge_labels
            }

            optimizer.zero_grad()

            # 🚀 使用 autocast 进行混合精度前向传播
            with autocast():
                if self.config.use_triple_consistency:
                    clean_output, perturbed_output = self.model.forward_with_feature_perturbation(
                        input_ids, attention_mask,
                        dropout_rate=self.config.feature_dropout_rate,
                        noise_std=self.config.feature_noise_std
                    )
                    loss, loss_dict = self.loss_calculator.compute_total_loss(
                        clean_output, labels, with_pcl=True, perturbed_output=perturbed_output
                    )
                else:
                    output = self.model(input_ids, attention_mask)
                    loss, loss_dict = self.loss_calculator.compute_total_loss(
                        output, labels, with_pcl=False
                    )

            # 🚀 使用 scaler 进行混合精度反向传播
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += loss.item()
            for key in loss_dict:
                if key in loss_stats:
                    loss_stats[key] += loss_dict[key]

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                self.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], self.global_step)

            self.global_step += 1

        avg_loss = total_loss / len(train_loader)
        for key in loss_stats:
            loss_stats[key] /= len(train_loader)

        return avg_loss, loss_stats

    def evaluate(self, data_loader, desc: str = "Evaluating"):
        """评估模型（使用混合精度）"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)

                # 🚀 评估时也使用混合精度
                with autocast():
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

        metrics = self.evaluator.compute_metrics(all_predictions, all_labels)
        align_ratio = self.evaluator.compute_alignment_ratio(
            all_predictions, all_labels, self.mapping_dict
        )
        metrics['alignment_ratio'] = align_ratio

        return metrics, all_predictions, all_labels

    def train(self, train_loader, dev_loader):
        """完整训练流程"""
        print("\n" + "="*60)
        print("🚀 Starting Training with Mixed Precision (AMP)")
        print("="*60)
        print(f"Device: {self.config.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Num workers: {self.config.num_workers}")
        print(f"Mixed Precision: FP16 ✓")
        print(f"Using cached data: {self.config.use_cached_data}")
        print("="*60 + "\n")

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

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*60}")

            train_loss, loss_stats = self.train_epoch(train_loader, optimizer, scheduler, epoch)

            print(f"\n📊 Epoch {epoch} Training Summary:")
            print(f"  Average Loss: {train_loss:.4f}")
            for key, value in loss_stats.items():
                if value > 0:
                    print(f"  {key}: {value:.4f}")

            print(f"\n🔍 Evaluating on development set...")
            dev_metrics, _, _ = self.evaluate(dev_loader, desc="Dev Eval")

            self.evaluator.print_metrics(dev_metrics, f"Epoch {epoch} Dev")

            for key, value in dev_metrics.items():
                self.writer.add_scalar(f'dev/{key}', value, epoch)

            current_metric = dev_metrics['avg_ma_f1']
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model saved! Avg Macro-F1: {current_metric:.4f}")

            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "="*60)
        print("✨ Training completed!")
        print(f"Best Avg Macro-F1: {self.best_metric:.4f}")
        print("="*60 + "\n")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def main():
    config = Config()

    print("📂 Loading data...")

    if config.use_cached_data and os.path.exists(config.cache_dir):
        print("✓ Using cached preprocessed data (Fast mode)")

        train_loader = get_fast_dataloader(
            config,
            os.path.join(config.cache_dir, 'train_cached.pt'),
            shuffle=True
        )

        dev_loader = get_fast_dataloader(
            config,
            os.path.join(config.cache_dir, 'dev_cached.pt'),
            shuffle=False
        )

        test_loader = get_fast_dataloader(
            config,
            os.path.join(config.cache_dir, 'test_cached.pt'),
            shuffle=False
        )
    else:
        print("❌ Cache not found. Please run: python preprocess_data.py")
        sys.exit(1)

    trainer = AMPTrainer(config)
    trainer.train(train_loader, dev_loader)

    best_model_path = os.path.join(config.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    print("\n🎯 Final evaluation on test set...")
    test_metrics, test_preds, test_labels = trainer.evaluate(test_loader, desc="Test Eval")
    trainer.evaluator.print_metrics(test_metrics, "Final Test")

    results = {
        'config': str(config),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'training_method': 'Mixed Precision (FP16)'
    }

    results_path = os.path.join(config.save_dir, 'test_results_amp.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Results saved to {results_path}")


if __name__ == "__main__":
    main()
