
import numpy as np
import os
import sys
# HumanParsing 모듈을 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import argparse
from torch.utils.data import DataLoader, Subset

# 현재 디렉토리 구조에 맞게 수정
from models import ParsingModel
from data import build_dataset, build_transforms, build_dataloader
from train import build_criterion, build_optimizer, build_scheduler, Trainer
from utils import Logger, Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Human Parsing Model')

    # 학습 모드 설정
    parser.add_argument('--mode-select', type=str, default='model',
                        help='choose your mode: model')

    # 기본 설정에 quick test 옵션 추가
    parser.add_argument('--quick-test', action='store_true', default=False,
                        help='Enable quick test mode with reduced dataset')
    # Basic configuration
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./model_outputs',
                        help='Directory to save outputs')

    # Data configuration - 전처리된 데이터 경로 사용
    parser.add_argument('--train-dir', type=str,
                        default="C:/Users/tjdwn/OneDrive/Desktop/parsingData/preprocessed/model/train",
                        help='Preprocessed training data directory for model')
    parser.add_argument('--val-dir', type=str,
                        default="C:/Users/tjdwn/OneDrive/Desktop/parsingData/preprocessed/model/val",
                        help='Preprocessed validation data directory for model')
    parser.add_argument('--mask-dir', type=str,
                        default="C:/Users/tjdwn/OneDrive/Desktop/parsingData/preprocessed/model",  # 예시
                        help='Directory containing pre-generated masks (if needed)')

    # Model configuration
    parser.add_argument('--backbone-pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--num-classes', type=int, default=21,
                        help='Number of classes')
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
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--gradient-clip-val', type=float, default=None,
                        help='Gradient clipping value')

    # Scheduler configuration
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Loss configuration
    parser.add_argument('--ce-weight', type=float, default=0.5,
                        help='Cross entropy loss weight')
    parser.add_argument('--dice-weight', type=float, default=2.0,
                        help='Dice loss weight')
    parser.add_argument('--focal-weight', type=float, default=0.5,
                        help='Focal loss weight')

    # Resume training
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    #추가 인자
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--early-stopping-delta', type=float, default=0.001,
                        help='Minimum change in monitored quantity to qualify as an improvement')
    parser.add_argument('--use-wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='human_parsing',
                        help='Weights & Biases project name')

    # Quick test 모드일 때 설정 자동 조정
    if args.quick_test:
        args.batch_size = 4
        args.epochs = 3
        args.gradient_accumulation_steps = 1
        args.num_workers = 4

    return args

#빠른 테스트를 위한 설정
def get_subset_indices(dataset_size, subset_size=1000):
    """Get random subset of indices"""
    indices = np.random.permutation(dataset_size)[:subset_size]
    return indices


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    logger = Logger(
        name='train',
        save_dir=args.output_dir
    )
    logger.info(f'Arguments: {args}')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Build transforms
    train_transform = build_transforms(is_train=True)
    val_transform = build_transforms(is_train=False)

    # Build datasets
    train_dataset = build_dataset(
        data_dir=args.train_dir,
        transforms=train_transform,
        split='train'
    )

    val_dataset = build_dataset(
        data_dir=args.val_dir,
        transforms=val_transform,
        split='val'
    )

    # Quick test 모드일 때 데이터셋 크기 제한
    if args.quick_test:
        logger.info("Using quick test mode with reduced dataset")
        train_dataset = Subset(
            train_dataset,
            get_subset_indices(len(train_dataset), 100)
        )
        val_dataset = Subset(
            val_dataset,
            get_subset_indices(len(val_dataset), 20)
        )

    # Build dataloaders
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False
    )

    # Build model
    model = ParsingModel(
        num_classes=args.num_classes,
        backbone_pretrained=args.backbone_pretrained,
        fpn_channels=args.fpn_channels,
        decoder_channels=args.decoder_channels
    )

    # Build criterion, optimizer, scheduler
    criterion = build_criterion(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )

    optimizer = build_optimizer(
        model=model,
        name='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_factor=0.1
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        name='onecycle',
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),  # 추가
        warmup_epochs=0,  # onecycle은 자체 warmup이 있으므로 0으로 설정
        max_lr=3e-4,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
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
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        resume_from=args.resume_from
    )

if __name__ == '__main__':
    main()