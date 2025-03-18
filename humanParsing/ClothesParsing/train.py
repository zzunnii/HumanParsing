import os
import sys
import json
import subprocess
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import argparse
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# 현재 디렉토리 구조에 맞게 수정
from .models import ParsingModel
from .data import build_dataset, build_transforms, build_dataloader
from .train import build_criterion, build_optimizer, build_scheduler, Trainer
from .utils import Logger, Visualizer
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train Human Parsing Model')

    # 학습 모드 설정 - 대분류 옵션 추가
    parser.add_argument('--mode-select', type=str, default='tops',
                        choices=['tops', 'bottoms'],
                        help='Choose training mode: model, tops, bottoms')

    # 기본 설정에 quick test 옵션 추가
    parser.add_argument('--quick-test', action='store_true', default=False,
                        help='Enable quick test mode with reduced dataset')

    # Basic configuration
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save model_output (auto-generated if None)')

    # Data configuration - 전처리된 데이터 경로 사용
    parser.add_argument('--data-root', type=str,
                        default="C:/Users/tjdwn/OneDrive/Desktop/parsingData/preprocessed",
                        help='Root directory of preprocessed data')

    # 원본 COCO annotation 파일 대신, 전처리 단계에서 생성한 person 구조를 사용하므로 annotation 파일은 더 이상 필요하지 않을 수 있음.
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Directory containing pre-generated masks (if needed)')

    # Model configuration
    parser.add_argument('--backbone-pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--backbone', type=str, default='resnet152',
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Backbone model')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-configured based on mode)')
    parser.add_argument('--fpn-channels', type=int, default=256,
                        help='Number of FPN channels')
    parser.add_argument('--decoder-channels', type=int, default=512,
                        help='Number of decoder channels')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Enable mixed precision')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--gradient-clip-val', type=float, default=None,
                        help='Gradient clipping value')

    # Scheduler configuration
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['cosine', 'step', 'onecycle'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Loss configuration
    parser.add_argument('--ce-weight', type=float, default=0.5,
                        help='Cross entropy loss weight')
    parser.add_argument('--dice-weight', type=float, default=2.0,
                        help='Dice loss weight')
    parser.add_argument('--focal-weight', type=float, default=1.0,
                        help='Focal loss weight')
    parser.add_argument('--class-weights', type=str, default=None,
                        help='Class weights as comma-separated values (e.g., "0.1,1.0,2.0")')

    # Resume training
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--early-stopping-delta', type=float, default=0.001,
                        help='Minimum change in monitored quantity to qualify as an improvement')

    # Wandb integration
    parser.add_argument('--use-wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='human_parsing',
                        help='Weights & Biases project name')

    # Data filtering
    parser.add_argument('--category-filter', type=str, default=None,
                        help='Filter specific category (e.g., "torso,rsleeve,lsleeve" for tops)')

    args = parser.parse_args()

    # 모드에 따라 자동 설정 구성
    configure_args_by_mode(args)

    # Quick test 모드일 때 설정 자동 조정
    if args.quick_test:
        args.batch_size = 4
        args.epochs = 100
        args.gradient_accumulation_steps = 1
        args.num_workers = 4

    return args


def configure_args_by_mode(args):

    if args.mode_select == 'tops':
        args.num_classes = 5
        args.train_dir = f"{args.data_root}/item/train"
        args.val_dir = f"{args.data_root}/item/val"

    elif args.mode_select == 'bottoms':
        args.num_classes = 7
        args.train_dir = f"{args.data_root}/item/train"
        args.val_dir = f"{args.data_root}/item/val"

    # 출력 디렉토리 자동 설정 (지정되지 않은 경우)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"output/{args.mode_select}_{timestamp}"

def calculate_class_weights(dataset, num_classes):
    """데이터셋에서 클래스별 픽셀 빈도를 계산해 가중치 생성"""
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    total_pixels = 0

    # 데이터셋의 모든 샘플에서 클래스 빈도 계산
    for i in range(len(dataset)):
        mask = dataset[i]['mask'].flatten()
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum().item()
        total_pixels += mask.numel()

    # 빈도가 0인 클래스에 대해 작은 값을 추가해 0으로 나누기 방지
    class_counts = class_counts + 1e-6
    # 클래스 빈도의 역수를 가중치로 사용
    class_weights = 1.0 / class_counts
    # 가중치를 정규화 (합이 num_classes가 되도록)
    class_weights = class_weights / class_weights.sum() * num_classes
    # args.class_weights 형식에 맞게 문자열로 변환
    return ",".join(map(str, class_weights.tolist()))

def get_subset_indices(dataset_size, subset_size=100):
    """Get random subset of indices"""
    indices = np.random.permutation(dataset_size)[:subset_size]
    return indices



def train_single_model(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(name='train', save_dir=args.output_dir)
    logger.info(f'Arguments: {args}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    train_transform = build_transforms(is_train=False)
    val_transform = build_transforms(is_train=False)

    # quick-test 모드에서 subset_size 설정
    train_subset_size = 1000 if args.quick_test else None
    val_subset_size = 100 if args.quick_test else None

    # 학습 데이터셋 생성
    train_dataset = build_dataset(
        data_dir=args.train_dir,
        transforms=train_transform,
        split='train',
        mode=args.mode_select,
        quick_test=args.quick_test,
        subset_size=train_subset_size,
        filter_empty=True
    )

    val_dataset = build_dataset(
        data_dir=args.val_dir,
        transforms=val_transform,
        split='val',
        mode=args.mode_select,
        quick_test=args.quick_test,
        subset_size=val_subset_size,
        filter_empty=True
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # 클래스 가중치 동적 계산
    if args.class_weights is None:  # 기본 가중치가 설정되지 않은 경우에만 계산
        args.class_weights = calculate_class_weights(train_dataset, args.num_classes)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=args.num_workers > 0
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=args.num_workers > 0
    )

    # Build model
    model = ParsingModel(
        num_classes=args.num_classes,
        backbone_pretrained=args.backbone_pretrained,
        fpn_channels=args.fpn_channels,
        decoder_channels=args.decoder_channels
    )

    # Build criterion
    criterion = build_criterion(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
    )

    optimizer = build_optimizer(
        model=model,
        name='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_factor=0.1
    )

    # 스케줄러 설정 (기존 코드 유지)
    if args.scheduler == 'onecycle':
        scheduler = build_scheduler(
            optimizer=optimizer,
            name='onecycle',
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            warmup_epochs=0,
            max_lr=3e-4,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
    else:
        scheduler = build_scheduler(
            optimizer=optimizer,
            name=args.scheduler,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr
        )

    visualizer = Visualizer(num_classes=args.num_classes)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        visualizer=visualizer,
        num_classes=args.num_classes,
        mixed_precision=args.mixed_precision,
        gradient_clip_val=args.gradient_clip_val,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        batch_size=args.batch_size,
        mode=args.mode_select
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        resume_from=args.resume_from
    )

    # 학습 완료 정보 저장
    training_info = {
        'mode': args.mode_select,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'final_metrics': trainer.best_metric,
        'completed': True
    }

    with open(os.path.join(args.output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)

    return training_info



def main():
    args = parse_args()
    train_single_model(args)


if __name__ == '__main__':
    main()