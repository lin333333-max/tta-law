"""
终极优化版训练脚本：
1. 混合精度（FP16）- 提速 2-3x
2. 矩阵化映射（去除循环）- 提速 10-50x
3. 矢量化损失计算 - 提速 5-10x
预计总体提速：5-10 倍
"""
import os
import sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
import json

from config import Config
from model import LegalJudgmentModel
from fast_data_loader import get_fast_dataloader
from data_loader_optimized import FastMappingDictionary  # 🚀 优化的映射
from loss_functions_optimized import FastLossCalculator  # 🚀 优化的损失
from evaluator import Evaluator


class UltraFastTrainer:
    """终极优化训练器（支持断点续跑）"""

    def __init__(self, config: Config, resume_from: str = None):
        self.config = config
        self._set_seed(config.seed)

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)

        # 🚀 使用优化的映射字典
        self.mapping_dict = FastMappingDictionary(
            config.a2c_dict_file,
            config.c2a_dict_file,
            num_articles=config.num_articles,
            num_charges=config.num_charges
        )

        self.model = LegalJudgmentModel(config).to(config.device)
        print(f"✓ Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # 🚀 使用优化的损失计算器
        self.loss_calculator = FastLossCalculator(config, self.mapping_dict)

        self.evaluator = Evaluator(threshold=config.threshold)
        self.writer = SummaryWriter(config.log_dir)

        # 混合精度
        self.scaler = GradScaler()

        # 训练状态
        self.start_epoch = 1
        self.global_step = 0
        self.best_metric = 0.0
        self.optimizer = None
        self.scheduler = None

        # 训练历史文件
        self.history_file = os.path.join(config.log_dir, 'training_history.json')
        self.training_history = []

        # 🔄 断点续跑：如果指定了resume路径，加载checkpoint
        if resume_from:
            self.resume_from_checkpoint(resume_from)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)

    def resume_from_checkpoint(self, checkpoint_path: str):
        """🔄 从checkpoint恢复训练"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint不存在: {checkpoint_path}")
            return

        print(f"\n{'='*60}")
        print(f"🔄 从checkpoint恢复训练: {checkpoint_path}")
        print(f"{'='*60}")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)

        # 恢复模型
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型权重已恢复")

        # 恢复训练状态
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)

        print(f"✓ 训练状态已恢复:")
        print(f"  - 起始Epoch: {self.start_epoch}")
        print(f"  - Global Step: {self.global_step}")
        print(f"  - Best Metric: {self.best_metric:.4f}")

        # 恢复训练历史
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
            print(f"✓ 训练历史已恢复 ({len(self.training_history)} epochs)")

        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            # 优化器会在train()中创建，这里先保存状态
            self._saved_optimizer_state = checkpoint['optimizer_state_dict']
            print(f"✓ 优化器状态待恢复")
        else:
            self._saved_optimizer_state = None

        # 恢复scheduler状态
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self._saved_scheduler_state = checkpoint['scheduler_state_dict']
            print(f"✓ Scheduler状态待恢复")
        else:
            self._saved_scheduler_state = None

        # 恢复scaler状态
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"✓ GradScaler状态已恢复")

        print(f"{'='*60}\n")

    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        loss_stats = {'ce_loss': 0.0, 'align_loss': 0.0, 'pcl_loss': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [ULTRA]")
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

            # 🚀 混合精度前向传播
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

            # 🚀 混合精度反向传播
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
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)

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

        # 使用优化的对齐度计算
        from loss_functions_optimized import FastAlignmentMetrics
        align_scores = []
        for pred, label in zip(all_predictions, all_labels):
            score = FastAlignmentMetrics.compute_sample_alignment(
                pred['articles'], pred['charges'],
                self.mapping_dict
            ).mean().item()
            align_scores.append(score)

        metrics['alignment_ratio'] = sum(align_scores) / len(align_scores)

        return metrics, all_predictions, all_labels

    def train(self, train_loader, dev_loader):
        """完整训练流程（支持断点续跑）"""
        print("\n" + "="*60)
        print("🚀 ULTRA-FAST Training")
        print("="*60)
        print(f"Device: {self.config.device}")
        print(f"Optimizations:")
        print(f"  ✓ Mixed Precision (FP16)")
        print(f"  ✓ Vectorized Mapping (No loops)")
        print(f"  ✓ Optimized Loss Calculation")
        print(f"  ✓ Cached Preprocessed Data")
        print(f"  ✓ Resumable Training")
        print(f"Start Epoch: {self.start_epoch}")
        print(f"Target Epoch: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Num workers: {self.config.num_workers}")
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

        # 🔄 恢复优化器和scheduler状态（如果有）
        if hasattr(self, '_saved_optimizer_state') and self._saved_optimizer_state:
            optimizer.load_state_dict(self._saved_optimizer_state)
            print("✓ 优化器状态已恢复")

        if hasattr(self, '_saved_scheduler_state') and self._saved_scheduler_state:
            scheduler.load_state_dict(self._saved_scheduler_state)
            print("✓ Scheduler状态已恢复")

        # 如果从checkpoint恢复且已有历史，使用已有的
        if not self.training_history and os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
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

            # 📝 保存本轮指标到历史记录
            epoch_record = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_loss_stats': loss_stats,
                'dev_metrics': {k: float(v) for k, v in dev_metrics.items()}
            }
            self.training_history.append(epoch_record)

            # 保存到文件（每轮都更新）
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)

            current_metric = dev_metrics['avg_ma_f1']
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(epoch, optimizer, scheduler, is_best=True)
                print(f"✅ New best model saved! Avg Macro-F1: {current_metric:.4f}")

            if epoch % 5 == 0:
                self.save_checkpoint(epoch, optimizer, scheduler, is_best=False)

        print("\n" + "="*60)
        print("✨ Training completed!")
        print(f"Best Avg Macro-F1: {self.best_metric:.4f}")
        print(f"📝 Training history saved to: {self.history_file}")
        print("="*60 + "\n")

    def save_checkpoint(self, epoch: int, optimizer, scheduler, is_best: bool = False):
        """保存checkpoint（包含完整训练状态，自动清理旧checkpoint）"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved to {best_path}")

        # 🔥 自动清理：只保留最近3个checkpoint和best_model.pt
        import glob
        all_checkpoints = sorted(glob.glob(os.path.join(self.config.save_dir, 'checkpoint_epoch_*.pt')))
        if len(all_checkpoints) > 3:
            for old_ckpt in all_checkpoints[:-3]:
                try:
                    os.remove(old_ckpt)
                    print(f"🗑️  自动删除旧checkpoint: {os.path.basename(old_ckpt)}")
                except Exception as e:
                    print(f"⚠️  删除失败: {old_ckpt}, {e}")

    def load_checkpoint(self, checkpoint_path: str):
        # weights_only=False：checkpoint 里存了 Config 对象，且来源是我们自己训练生成的，可信
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练脚本（支持断点续跑）')
    parser.add_argument('--resume', type=str, default=None,
                       help='从checkpoint恢复训练的路径（例如：./checkpoints/checkpoint_epoch_10.pt）')
    parser.add_argument('--test-only', action='store_true',
                       help='跳过训练，仅加载 best_model.pt 在 test 集上评估')
    args = parser.parse_args()

    config = Config()

    print("📂 Loading data...")

    if not (config.use_cached_data and os.path.exists(config.cache_dir)):
        print("❌ Cache not found. Please run: python preprocess_data.py")
        sys.exit(1)

    print("✓ Using cached preprocessed data")

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

    # 🔄 创建训练器（支持断点续跑）
    trainer = UltraFastTrainer(config, resume_from=args.resume)

    best_model_path = os.path.join(config.save_dir, 'best_model.pt')

    if args.test_only:
        # 跳过训练，直接加载已有的最佳模型做 test 集评估
        if not os.path.exists(best_model_path):
            print(f"❌ 找不到 {best_model_path}，无法执行 --test-only")
            sys.exit(1)
        trainer.load_checkpoint(best_model_path)
    else:
        trainer.train(train_loader, dev_loader)
        if os.path.exists(best_model_path):
            trainer.load_checkpoint(best_model_path)

    print("\n🎯 Final evaluation on test set...")
    test_metrics, test_preds, test_labels = trainer.evaluate(test_loader, desc="Test Eval")
    trainer.evaluator.print_metrics(test_metrics, "Final Test")

    results = {
        'config': str(config),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'optimizations': [
            'Mixed Precision (FP16)',
            'Vectorized Mapping',
            'Optimized Loss Calculation',
            'Cached Preprocessed Data'
        ]
    }

    results_path = os.path.join(config.save_dir, 'test_results_ultrafast.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Results saved to {results_path}")


if __name__ == "__main__":
    main()

