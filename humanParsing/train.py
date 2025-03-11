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
from parsing.models import ParsingModel
from parsing.data import build_dataset, build_transforms, build_dataloader
from parsing.train import build_criterion, build_optimizer, build_scheduler, Trainer
from parsing.utils import Logger, Visualizer
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train Human Parsing Model')

    # 학습 모드 설정 - 대분류 옵션 추가
    parser.add_argument('--mode-select', type=str, default='models',
                        choices=['model', 'tops', 'bottoms'],
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
                        default="./parsingData/rawdata",
                        help='Root directory of rawdata data')

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
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-5,
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
        args.epochs = 3
        args.gradient_accumulation_steps = 1
        args.num_workers = 4

    return args


def configure_args_by_mode(args):
    """모드에 따라 인자 자동 설정"""
    # 모드별 클래스 수 설정
    if args.mode_select == 'model':
        args.num_classes = 20
        args.train_dir = f"{args.data_root}/model/train"
        args.val_dir = f"{args.data_root}/model/val"
    elif args.mode_select == 'tops':
        args.num_classes = 5  # 배경 + 모자 + 소매 + 몸통
        args.train_dir = f"{args.data_root}/item/train"
        args.val_dir = f"{args.data_root}/item/val"
    elif args.mode_select == 'bottoms':
        args.num_classes = 7  # 배경 + 엉덩이 + 바지 + 스커트
        args.train_dir = f"{args.data_root}/item/train"
        args.val_dir = f"{args.data_root}/item/val"

    # 출력 디렉토리 자동 설정 (지정되지 않은 경우)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"output/{args.mode_select}_{timestamp}"

    # 클래스 가중치 설정
    if args.class_weights is None:
        if args.mode_select == 'tops':
            args.class_weights = "0.1,2.0,2.0,2.0"  # 배경,모자,소매,몸통
        elif args.mode_select == 'bottoms':
            args.class_weights = "0.1,2.0,2.0,2.0"  # 배경,엉덩이,바지,스커트
        elif args.mode_select == 'model':
            # 배경 가중치 낮게, 나머지 클래스 가중치 높게
            weights = ["0.1"] + ["2.0"] * 19
            args.class_weights = ",".join(weights)


def get_subset_indices(dataset_size, subset_size=1000):
    """Get random subset of indices"""
    indices = np.random.permutation(dataset_size)[:subset_size]
    return indices


def get_class_weights(class_weights_str, num_classes):
    """문자열 형태의 클래스 가중치를 파싱하여 텐서로 변환"""
    if class_weights_str is None:
        return None

    weights = [float(w) for w in class_weights_str.split(',')]

    # 가중치 개수가 클래스 수와 다르면 경고 출력
    if len(weights) != num_classes:
        print(f"Warning: Number of class weights ({len(weights)}) does not match number of classes ({num_classes})")

        # 필요하면 가중치를 확장하거나 축소
        if len(weights) < num_classes:
            # 부족한 가중치는 마지막 가중치로 채움
            weights.extend([weights[-1]] * (num_classes - len(weights)))
        else:
            # 초과 가중치는 잘라냄
            weights = weights[:num_classes]

    return torch.tensor(weights)


def train_single_model(args):
    """단일 모델 학습 함수"""
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
        split='train',
        mode=args.mode_select
    )

    val_dataset = build_dataset(
        data_dir=args.val_dir,
        transforms=val_transform,
        split='val',
        mode=args.mode_select
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
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0
    )
    for i in range(min(5, len(val_dataset))):
        sample = val_dataset[i]
        mask = sample['mask']
        unique_classes = torch.unique(mask)
        print(f"Sample {i} contains classes: {unique_classes.tolist()}")
    # Build model
    model = ParsingModel(
        num_classes=args.num_classes,
        backbone_pretrained=args.backbone_pretrained,
        fpn_channels=args.fpn_channels,
        decoder_channels=args.decoder_channels
    )

    # Parse class weights
    class_weights = get_class_weights(args.class_weights, args.num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)  # 디바이스로 이동

    # Build criterion, optimizer, scheduler
    criterion = build_criterion(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        class_weights=class_weights.tolist() if class_weights is not None else None
    )

    optimizer = build_optimizer(
        model=model,
        name='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_factor=0.1
    )

    # 스케줄러 선택
    if args.scheduler == 'onecycle':
        scheduler = build_scheduler(
            optimizer=optimizer,
            name='onecycle',
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            warmup_epochs=0,  # onecycle은 자체 warmup이 있으므로 0으로 설정
            max_lr=args.lr * 10,
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

    # 시각화 도구 설정
    visualizer = Visualizer(num_classes=args.num_classes)

    # 트레이너 설정
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
        mode=args.mode_select  # 현재 선택된 모드 전달
    )

    # 학습 실행
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


def train_all_categories(base_args):
    """모든 카테고리 모델을 순차적으로 학습"""
    categories = ['tops', 'bottoms', 'shoes']
    results = {}

    # 각 카테고리별 학습 실행
    for category in categories:
        print(f"\n{'=' * 50}")
        print(f"Training {category.upper()} model...")
        print(f"{'=' * 50}\n")

        # 새로운 args 객체 생성 (기존 설정 유지하면서 카테고리만 변경)
        category_args = argparse.Namespace(**vars(base_args))
        category_args.mode_select = category

        # 카테고리별 출력 디렉토리 설정
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        category_args.output_dir = f"output/{category}_{timestamp}"

        # 카테고리에 맞게 설정 자동 구성
        configure_args_by_mode(category_args)

        # 학습 실행
        try:
            result = train_single_model(category_args)
            results[category] = result
        except Exception as e:
            print(f"Error training {category} model: {e}")
            results[category] = {'error': str(e), 'completed': False}

    # 학습 결과 요약
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)

    for category, result in results.items():
        if result.get('completed', False):
            print(f"{category.upper()}: Best metric = {result.get('final_metrics', 'N/A')}")
        else:
            print(f"{category.upper()}: Failed - {result.get('error', 'Unknown error')}")

    # 결과 저장
    with open('auto_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    args = parse_args()

    if args.mode_select == 'auto':
        # 자동 모드: 모든 카테고리 순차 학습
        train_all_categories(args)
    else:
        # 일반 모드: 단일 모델 학습
        train_single_model(args)


if __name__ == '__main__':
    main()