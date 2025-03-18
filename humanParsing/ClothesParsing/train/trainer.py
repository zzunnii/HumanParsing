import os
import time
import cv2
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler  # GradScaler는 여기서 가져옴
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from ..data import DataPrefetcher
from ..utils import (
    AverageMeter,
    SegmentationMetric,
    TensorboardLogger,
    Logger,
    Visualizer
)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            device: torch.device = torch.device('cuda'),
            output_dir: str = 'model_output',
            visualizer: Optional[Visualizer] = None,
            num_classes: int = 19,
            mixed_precision: bool = True,
            gradient_clip_val: Optional[float] = None,
            gradient_accumulation_steps: int = 1,
            print_freq: int = 10,
            save_freq: int = 5,
            early_stopping_patience: int = 10,
            early_stopping_delta: float = 0.001,
            use_wandb: bool = True,
            wandb_project: str = "human_parsing",
            batch_size: int = 32,
            mode: str = "model"
    ):
        # 기본 모델 및 최적화 설정
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.visualizer = visualizer
        self.mode = mode
        self.num_classes = num_classes  # num_classes도 초기화

        # 학습 설정
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.print_freq = print_freq
        self.save_freq = save_freq

        # Early stopping 설정
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_counter = 0
        self.best_score = None

        # 메트릭 및 로깅 설정
        self.metric = SegmentationMetric(num_classes=num_classes, device=device)
        os.makedirs(output_dir, exist_ok=True)
        self.tb_logger = TensorboardLogger(os.path.join(output_dir, 'tensorboard'))
        self.logger = Logger(name='train', save_dir=output_dir)

        # Mixed precision 설정 (FutureWarning 해결)
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        self.best_metric = 0.0
        self.global_step = 0

        # Wandb 설정 (use_wandb 먼저 정의)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project  # wandb_project도 저장
        if self.use_wandb:
            try:
                import wandb
                self.logger.info("Initializing WandB...")
                if wandb.run is None:
                    wandb.init(project=wandb_project, config={
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "batch_size": batch_size,
                        "mixed_precision": mixed_precision,
                        "model": model.__class__.__name__,
                        "early_stopping_patience": early_stopping_patience,
                        "gradient_accumulation_steps": gradient_accumulation_steps
                    })
                self.logger.info("WandB initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize WandB: {str(e)}")
                self.use_wandb = False

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.metric.reset()
        prefetcher = DataPrefetcher(train_loader, self.device)
        batch = prefetcher.next()
        idx = 0
        end = time.time()
        self.optimizer.zero_grad()

        while batch is not None:
            data_time.update(time.time() - end)
            images = batch['image']
            targets = batch['mask']

            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip_val is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            predictions = torch.argmax(outputs, dim=1)
            self.metric.update(predictions, targets)
            losses.update(loss.item() * self.gradient_accumulation_steps)
            batch_time.update(time.time() - end)
            batch_iou = self.metric.get_scores()

            if idx % self.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.logger.info(
                    f'Epoch: [{epoch}][{idx}/{len(train_loader)}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                    f'mIoU {batch_iou["mean_iu"]:.4f} '
                    f'Pixel Acc {batch_iou["pixel_acc"]:.4f} '
                    f'LR {current_lr:.2e}'
                )
                self.tb_logger.log_scalar('train/loss_step', losses.val, self.global_step)
                self.tb_logger.log_scalar('train/lr', current_lr, self.global_step)
                self.tb_logger.log_scalar('train/memory', memory_used, self.global_step)
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "train/loss_step": losses.val,
                            "train/lr": current_lr,
                            "train/memory": memory_used,
                            "train/batch_miou": batch_iou["mean_iu"],
                            "train/batch_pixel_acc": batch_iou["pixel_acc"]
                        }, step=self.global_step)
                    except Exception as e:
                        self.logger.error(f"Failed to log metrics to WandB: {str(e)}")
                self.global_step += 1

            batch = prefetcher.next()
            idx += 1
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        scores = self.metric.get_scores()
        return scores

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        self.metric.reset()
        end = time.time()

        for i, batch in enumerate(val_loader):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            predictions = torch.argmax(outputs, dim=1)
            self.metric.update(predictions, targets)

            losses.update(loss.item())
            batch_time.update(time.time() - end)

            if self.visualizer is not None and i == 0:
                vis_img = self.visualizer.visualize_prediction(
                    images[0].cpu(),
                    predictions[0].cpu(),
                    targets[0].cpu()
                )
                if vis_img is not None and vis_img.ndim == 3 and vis_img.shape[-1] == 3:
                    self.tb_logger.log_image('val/predictions', vis_img, self.global_step)

                class_vis = self.visualizer.visualize_classes_separately(
                    images[0].cpu(),
                    predictions[0].cpu(),
                    alpha=0.7,
                    max_classes=self.visualizer.num_classes
                )
                if class_vis is not None:
                    save_path = os.path.join(self.output_dir, f'class_vis_epoch{epoch}_batch{i}.jpg')
                    cv2.imwrite(save_path, cv2.cvtColor(class_vis, cv2.COLOR_RGB2BGR))
                    self.tb_logger.log_image('val/class_predictions', class_vis, self.global_step)
                    if self.use_wandb:
                        try:
                            import wandb
                            wandb.log({"val/class_predictions": wandb.Image(class_vis)}, step=self.global_step)
                        except Exception as e:
                            self.logger.error(f"Failed to log class image to WandB: {str(e)}")

            end = time.time()

        scores = self.metric.get_scores()
        self.tb_logger.log_scalar('val/loss', losses.avg, self.global_step)
        self.tb_logger.log_scalar('val/miou', scores['mean_iu'], self.global_step)
        self.tb_logger.log_scalar('val/pixel_acc', scores['pixel_acc'], self.global_step)

        self.logger.info(
            f'Validation Results - Epoch: [{epoch}]\n'
            f'Loss {losses.avg:.4f}\n'
            f'mIoU {scores["mean_iu"]:.4f}\n'
            f'Pixel Acc {scores["pixel_acc"]:.4f}'
        )
        return scores

    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'metric': metric
        }
        save_path = os.path.join(self.output_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(state, save_path)
        if is_best:
            best_path = os.path.join(self.output_dir, 'model_best.pth')
            torch.save(state, best_path)

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            resume_from: Optional[str] = None
    ):
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.resume(resume_from)

        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f'Starting epoch {epoch}')
            train_scores = self.train_epoch(train_loader, epoch)
            val_scores = self.validate(val_loader, epoch)

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/miou": train_scores["mean_iu"],
                        "train/pixel_acc": train_scores["pixel_acc"],
                        "val/loss": val_scores.get("loss", 0),
                        "val/miou": val_scores["mean_iu"],
                        "val/pixel_acc": val_scores["pixel_acc"],
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
                except Exception as e:
                    self.logger.error(f"Failed to log epoch metrics to WandB: {str(e)}")

            current_score = val_scores['mean_iu']
            if self.best_score is None:
                self.best_score = current_score
            elif current_score > self.best_score + self.early_stopping_delta:
                self.best_score = current_score
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                self.logger.info(
                    f'EarlyStopping counter: {self.early_stopping_counter} out of {self.early_stopping_patience}')
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info('Early stopping triggered')
                    break

            is_best = val_scores['mean_iu'] > self.best_metric
            if is_best:
                self.best_metric = val_scores['mean_iu']
            self.save_checkpoint(epoch, val_scores['mean_iu'], is_best)

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                self.logger.error(f"Failed to finish WandB run: {str(e)}")

    def resume(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_metric = checkpoint['metric']
        return checkpoint['epoch'] + 1