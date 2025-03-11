#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모든 카테고리 모델을 자동으로 학습하는 스크립트
실행 방법: python train_all_categories.py
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Train all category models')

    # 각 카테고리 모델 활성화/비활성화
    parser.add_argument('--train-tops', action='store_true', default=True,
                        help='Train tops model')
    parser.add_argument('--train-bottoms', action='store_true', default=True,
                        help='Train bottoms model')
    parser.add_argument('--train-shoes', action='store_true', default=True,
                        help='Train shoes model')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--quick-test', action='store_true', default=False,
                        help='Run in quick test mode')

    return parser.parse_args()


def train_category(category, epochs, batch_size, lr, quick_test=False):
    """단일 카테고리 모델 학습"""
    print(f"\n{'=' * 50}")
    print(f"Training {category.upper()} model...")
    print(f"{'=' * 50}\n")

    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{category}_{timestamp}"

    # train.py 실행 명령 구성
    cmd = [
        sys.executable, "train.py",
        "--mode-select", category,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--use-wandb"
    ]

    # 빠른 테스트 모드 활성화
    if quick_test:
        cmd.append("--quick-test")

    # 모델 학습 실행
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        success = result.returncode == 0

        # 학습 결과 반환
        return {
            "category": category,
            "output_dir": output_dir,
            "success": success,
            "timestamp": timestamp
        }
    except subprocess.CalledProcessError as e:
        print(f"Error training {category} model: {e}")
        return {
            "category": category,
            "output_dir": output_dir,
            "success": False,
            "error": str(e),
            "timestamp": timestamp
        }


def main():
    args = parse_args()
    results = []

    # 카테고리 목록 생성
    categories = []
    if args.train_tops:
        categories.append("tops")
    if args.train_bottoms:
        categories.append("bottoms")
    if args.train_shoes:
        categories.append("shoes")

    # 각 카테고리별로 학습 실행
    for category in categories:
        result = train_category(
            category=category,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            quick_test=args.quick_test
        )
        results.append(result)

    # 결과 요약
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"{result['category'].upper()}: {status} - Output dir: {result['output_dir']}")

    # 결과 저장
    with open('training_results.json', 'w') as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }, f, indent=2)

    # 성공/실패 개수 계산
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    print(f"\nTraining complete: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    main()