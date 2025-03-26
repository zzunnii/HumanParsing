# test.py 수정 버전
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import ParsingModel
from data import build_dataset, build_transforms, build_dataloader
from utils import Logger, Visualizer, SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Test Human Parsing Model')

    # Basic configuration
    parser.add_argument('--test-dir', type=str, default="./parsingData/preprocessed/item/test",
                        help='Preprocessed test data directory')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    # Model configuration
    parser.add_argument('--num-classes', type=int, default=0,
                        help='Number of classes (0 for auto-detect based on mode)')
    parser.add_argument('--fpn-channels', type=int, default=256,
                        help='Number of FPN channels')
    parser.add_argument('--decoder-channels', type=int, default=512,
                        help='Number of decoder channels')

    # Test configuration
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-images', action='store_true',
                        help='Save visualization of predictions')
    parser.add_argument('--mode-select', type=str, default='tops',
                        choices=['tops', 'bottoms'],
                        help='Choose mode: model, tops, or bottoms')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    logger = Logger(
        name='test',
        save_dir=args.output_dir
    )
    logger.info(f'Arguments: {args}')

    if args.mode_select == 'tops':
        num_classes = 5
        checkpoint = args.checkpoint or r".\ClothesParsing\output\tops_20250326_122103\checkpoint_epoch5.pth"
    else:  # bottoms
        num_classes = 7
        checkpoint = args.checkpoint or r".\ClothesParsing\output\bottoms_20250326_182611\checkpoint_epoch7.pth"

    # 명령줄에서 지정된 값이 있으면 우선 사용
    if args.num_classes > 0:
        num_classes = args.num_classes

    # Build transforms
    transform = build_transforms(is_train=False)

    # Build dataset
    test_dataset = build_dataset(
        data_dir=args.test_dir,
        transforms=transform,
        split='test',
        mode=args.mode_select,  # 모드 전달
        filter_empty=True       # 필터링 활성화
    )

    # Build dataloader
    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # Build model
    model = ParsingModel(
        num_classes=num_classes,
        backbone_pretrained=False,  # 불필요한 pretrained 로딩 방지
        fpn_channels=args.fpn_channels,
        decoder_channels=args.decoder_channels
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['state_dict'])

    # 명시적으로 모델을 device로 이동
    model = model.to(device)
    logger.info(f"Loaded checkpoint successfully")

    # Initialize metric
    metric = SegmentationMetric(num_classes=num_classes)

    # Initialize visualizer
    visualizer = Visualizer(num_classes=num_classes)

    # Run test
    model.eval()
    metric.reset()

    # Create progress bar
    pbar = tqdm(total=len(test_loader), desc=f'Testing {args.mode_select} mode')

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # Get batch data
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)

            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Update metric
            metric.update(predictions, targets)

            # Save visualizations
            if args.save_images:
                for i in range(len(images)):
                    vis_img = visualizer.visualize_prediction(
                        images[i], predictions[i], targets[i]
                    )
                    save_path = os.path.join(
                        args.output_dir,
                        'visualizations',
                        f'{args.mode_select}_pred_{idx * len(images) + i}.png'
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    visualizer.save_visualization(vis_img, save_path)

            pbar.update(1)

        pbar.close()

    # Get final metrics
    scores = metric.get_scores()

    # Log results
    logger.info(f'\n{args.mode_select.capitalize()} Test Results:')
    logger.info(f'Mean IoU: {scores["mean_iu"]:.4f}')
    logger.info(f'Pixel Accuracy: {scores["pixel_acc"]:.4f}')
    logger.info(f'Mean F1: {scores["mean_f1"]:.4f}')

    # 클래스별 결과 (있는 경우)
    if "class_iu" in scores:
        logger.info("\nClass IoU:")
        for cls_idx, iou in enumerate(scores["class_iu"]):
            logger.info(f"Class {cls_idx}: {iou:.4f}")

    # Save detailed results
    import json
    with open(os.path.join(args.output_dir, f'{args.mode_select}_test_results.json'), 'w') as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.float32, np.float64)) else
               [float(x) for x in v] if isinstance(v, np.ndarray) else v
             for k, v in scores.items()},
            f,
            indent=4
        )

    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()