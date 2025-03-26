import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import defaultdict


class MaskAnalyzer:
    """Class for analyzing mask data statistics and object placement"""

    def __init__(self, data_dir, mode='item', output_dir='analysis_results', filter_type=None):
        """
        Args:
            data_dir: Dataset path (parent directory containing train, val folders)
            mode: 'item' only
            output_dir: Path to save analysis results
            filter_type: 'tops', 'bottoms', or None (no filtering)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.filter_type = filter_type

        # 필터링 타입에 따라 출력 디렉토리 조정
        if filter_type:
            self.output_dir = os.path.join(output_dir, filter_type)
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # 필터 타입에 따른 카테고리 맵핑 설정
        self.top_categories = {
            "background": 0,
            "rsleeve": 1,
            "lsleeve": 2,
            "torso": 3,
            "top_hidden": 4
        }

        self.bottom_categories = {
            "background": 0,
            "hip": 1,
            "pants_rsleeve": 2,
            "pants_lsleeve": 3,
            "pants_hidden": 4,
            "skirt": 5,
            "skirt_hidden": 6
        }

        # Variables to store analysis results
        self.person_ratios = []  # Person area ratio compared to image
        self.person_centers = []  # Person center coordinates (normalized)
        self.person_bboxes = []  # Person bounding boxes (normalized)
        self.person_heights = []  # Person height ratios
        self.person_widths = []  # Person width ratios
        self.person_areas = []  # Person pixel areas
        self.total_areas = []  # Total image areas

        # Class pixel statistics
        self.class_pixel_counts = defaultdict(int)
        self.total_pixels = 0

        # Color map for mask visualization
        self.colormap = plt.cm.get_cmap('tab20', 21)  # Colors for 21 classes

    def analyze_dataset(self, max_samples=None):
        """Analyze the entire dataset

        Args:
            max_samples: Maximum number of samples to analyze (None=all)
        """
        # List all person directories
        person_dirs = []
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            if os.path.exists(split_dir):
                sub_dirs = [os.path.join(split_dir, d) for d in os.listdir(split_dir)
                            if os.path.isdir(os.path.join(split_dir, d))]
                person_dirs.extend(sub_dirs)

        print(f"Found {len(person_dirs)} person directories in total")

        # Limit maximum number of samples
        if max_samples is not None and max_samples < len(person_dirs):
            import random
            random.shuffle(person_dirs)
            person_dirs = person_dirs[:max_samples]
            print(f"Randomly selected {max_samples} samples")

        # Analyze each person directory
        valid_samples = 0
        for person_dir in tqdm(person_dirs, desc=f"Analyzing {self.filter_type if self.filter_type else 'All'} Data"):
            if self.analyze_person(person_dir):
                valid_samples += 1

        print(f"Successfully analyzed {valid_samples} valid samples")

        # Calculate and visualize overall statistics
        self.calculate_statistics()

    def should_include_sample(self, mask_info):
        """필터 타입에 따라 샘플 포함 여부 결정"""
        if not self.filter_type:
            return True  # 필터링 없음

        mask_entries = mask_info.get("item_masks", [])

        # 필터링을 위한 플래그
        has_target_category = False

        for mask_entry in mask_entries:
            product_type = mask_entry.get("product_type", "")

            if self.filter_type == 'tops':
                if product_type in self.top_categories:
                    has_target_category = True
                    break
            elif self.filter_type == 'bottoms':
                if product_type in self.bottom_categories:
                    has_target_category = True
                    break

        return has_target_category

    def analyze_person(self, person_dir):
        """Analyze a single person directory"""
        try:
            # Load image
            image_path = os.path.join(person_dir, "item_image.jpg")
            if not os.path.exists(image_path):
                return False  # Skip if image doesn't exist

            image = cv2.imread(image_path)
            if image is None:
                return False

            H, W = image.shape[:2]

            # Load mask info
            mask_info_path = os.path.join(person_dir, "mask_info.json")
            if not os.path.exists(mask_info_path):
                return False

            with open(mask_info_path, 'r', encoding='utf-8') as f:
                mask_info = json.load(f)

            # 필터링 적용
            if not self.should_include_sample(mask_info):
                return False

            total_image_area = H * W
            self.total_areas.append(total_image_area)

            # Combine masks (all classes)
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            any_mask = np.zeros((H, W), dtype=np.bool_)  # Binary mask (all classes)

            # Process each mask entry
            mask_entries = mask_info.get("item_masks", [])
            for mask_entry in mask_entries:
                # 필터링 적용 - tops/bottoms 모드에 맞는 카테고리만 처리
                product_type = mask_entry.get("product_type", "unknown")

                if self.filter_type == 'tops' and product_type not in self.top_categories:
                    continue
                elif self.filter_type == 'bottoms' and product_type not in self.bottom_categories:
                    continue

                # 카테고리에 맞는 class_id 가져오기
                if self.filter_type == 'tops':
                    class_id = self.top_categories.get(product_type, 0)
                elif self.filter_type == 'bottoms':
                    class_id = self.bottom_categories.get(product_type, 0)
                else:
                    class_id = 1  # 필터링 없으면 기본값 1 (배경 외 다른 클래스)

                mask_path = mask_entry.get("path", None)
                if mask_path and os.path.exists(mask_path):
                    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if binary_mask.shape != (H, W):
                        binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                    # Set class ID in the combined mask
                    combined_mask[binary_mask > 127] = class_id
                    any_mask[binary_mask > 127] = True

                    # Count pixels per class
                    pixel_count = np.sum(binary_mask > 127)
                    self.class_pixel_counts[product_type] += pixel_count

            # Add to total pixel count
            self.total_pixels += H * W

            # Skip if no foreground content found
            if not np.any(any_mask):
                return False

            # Calculate item area in pixels
            item_pixel_area = np.sum(any_mask)
            self.person_areas.append(item_pixel_area)

            # Calculate item area ratio
            item_ratio = item_pixel_area / total_image_area
            self.person_ratios.append(item_ratio)

            # Calculate item boundaries
            rows = np.any(any_mask, axis=1)
            cols = np.any(any_mask, axis=0)

            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                # Normalize bounding box (0-1 range)
                bbox = [x_min / W, y_min / H, x_max / W, y_max / H]
                self.person_bboxes.append(bbox)

                # Calculate center point (normalized)
                center_x = ((x_min + x_max) / 2) / W
                center_y = ((y_min + y_max) / 2) / H
                self.person_centers.append((center_x, center_y))

                # Calculate height and width ratios
                height_ratio = (y_max - y_min) / H
                width_ratio = (x_max - x_min) / W
                self.person_heights.append(height_ratio)
                self.person_widths.append(width_ratio)

            return True  # 성공적으로 분석됨

        except Exception as e:
            print(f"Analysis error ({person_dir}): {str(e)}")
            return False

    def calculate_statistics(self):
        """Calculate and visualize statistics from collected data"""
        if not self.person_ratios:
            print("No data to analyze.")
            return

        # 1. Item ratio histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.person_ratios, bins=20, alpha=0.7)
        title_prefix = f"{self.filter_type.capitalize()} " if self.filter_type else ""
        plt.title(f'{title_prefix}Area Ratio Distribution')
        plt.xlabel('Area Ratio (0-1)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(self.output_dir, f'{self.filter_type if self.filter_type else "all"}_ratio_histogram.png'))

        # 2. Item center point distribution
        plt.figure(figsize=(10, 8))
        centers = np.array(self.person_centers)
        plt.scatter(centers[:, 0], centers[:, 1], alpha=0.5, s=10)
        plt.title(f'{title_prefix}Center Distribution')
        plt.xlabel('X Coordinate (0-1)')
        plt.ylabel('Y Coordinate (0-1)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.savefig(
            os.path.join(self.output_dir, f'{self.filter_type if self.filter_type else "all"}_center_distribution.png'))

        # 3. Item height/width ratio scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.person_widths, self.person_heights, alpha=0.5, s=10)
        plt.title(f'{title_prefix}Size Ratio Distribution')
        plt.xlabel('Width Ratio (0-1)')
        plt.ylabel('Height Ratio (0-1)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(self.output_dir, f'{self.filter_type if self.filter_type else "all"}_size_distribution.png'))

        # 4. Class pixel ratios
        class_ratios = {k: v / self.total_pixels for k, v in self.class_pixel_counts.items()}
        plt.figure(figsize=(12, 6))
        plt.bar(class_ratios.keys(), class_ratios.values())
        plt.title(f'{title_prefix}Class Pixel Ratios')
        plt.xlabel('Class')
        plt.ylabel('Ratio (0-1)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'{self.filter_type if self.filter_type else "all"}_class_pixel_ratios.png'))

        # 5. Item area distribution (in pixels)
        plt.figure(figsize=(10, 6))
        plt.hist(self.person_areas, bins=30, alpha=0.7)
        plt.title(f'{title_prefix}Area Distribution (in pixels)')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(self.output_dir, f'{self.filter_type if self.filter_type else "all"}_area_distribution.png'))

        # 6. Statistics summary
        summary = {
            'filter_type': self.filter_type,
            'sample_count': len(self.person_ratios),
            'area_ratio': {
                'mean': float(np.mean(self.person_ratios)),
                'median': float(np.median(self.person_ratios)),
                'min': float(np.min(self.person_ratios)),
                'max': float(np.max(self.person_ratios))
            },
            'height': {
                'mean': float(np.mean(self.person_heights)),
                'median': float(np.median(self.person_heights)),
                'min': float(np.min(self.person_heights)),
                'max': float(np.max(self.person_heights))
            },
            'width': {
                'mean': float(np.mean(self.person_widths)),
                'median': float(np.median(self.person_widths)),
                'min': float(np.min(self.person_widths)),
                'max': float(np.max(self.person_widths))
            },
            'area_pixels': {
                'mean': float(np.mean(self.person_areas)),
                'median': float(np.median(self.person_areas)),
                'min': float(np.min(self.person_areas)),
                'max': float(np.max(self.person_areas))
            },
            'center_x': {
                'mean': float(np.mean([c[0] for c in self.person_centers])),
                'median': float(np.median([c[0] for c in self.person_centers]))
            },
            'center_y': {
                'mean': float(np.mean([c[1] for c in self.person_centers])),
                'median': float(np.median([c[1] for c in self.person_centers]))
            },
            'class_ratios': class_ratios
        }

        # Save statistics
        with open(os.path.join(self.output_dir,
                               f'{self.filter_type if self.filter_type else "all"}_statistics_summary.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\n--- {title_prefix}Statistics Summary ---")
        print(f"Total samples analyzed: {summary['sample_count']}")
        print(f"Area ratio - Mean: {summary['area_ratio']['mean']:.4f}, Median: {summary['area_ratio']['median']:.4f}")
        print(f"Height - Mean: {summary['height']['mean']:.4f}, Median: {summary['height']['median']:.4f}")
        print(f"Width - Mean: {summary['width']['mean']:.4f}, Median: {summary['width']['median']:.4f}")
        print(f"Center X - Mean: {summary['center_x']['mean']:.4f}, Median: {summary['center_x']['median']:.4f}")
        print(f"Center Y - Mean: {summary['center_y']['mean']:.4f}, Median: {summary['center_y']['median']:.4f}")
        print(f"Area (pixels) - Mean: {summary['area_pixels']['mean']:.1f}, Median: {summary['area_pixels']['median']:.1f}")
        print("--------------------------------")

        return summary

    def visualize_sample_masks(self, num_samples=5):
        """Visualize masks from random samples"""
        # List person directories
        person_dirs = []
        for split in ['train', 'val']:
            split_dir = os.path.join(self.data_dir, split)
            if os.path.exists(split_dir):
                sub_dirs = [os.path.join(split_dir, d) for d in os.listdir(split_dir)
                            if os.path.isdir(os.path.join(split_dir, d))]
                person_dirs.extend(sub_dirs)

        # 필터링에 맞는 샘플만 선택
        filtered_samples = []
        for person_dir in tqdm(person_dirs[:min(100, len(person_dirs))],
                               desc="Finding valid samples for visualization"):
            try:
                mask_info_path = os.path.join(person_dir, "mask_info.json")
                if os.path.exists(mask_info_path):
                    with open(mask_info_path, 'r', encoding='utf-8') as f:
                        mask_info = json.load(f)

                    if self.should_include_sample(mask_info):
                        filtered_samples.append(person_dir)
                        if len(filtered_samples) >= num_samples * 2:  # 충분한 수의 샘플 확보
                            break
            except:
                continue

        # 랜덤 샘플 선택
        import random
        if filtered_samples:
            random.shuffle(filtered_samples)
            samples = filtered_samples[:num_samples]
        else:
            print("No valid samples found for visualization")
            return

        for i, person_dir in enumerate(samples):
            try:
                # Load image
                image_path = os.path.join(person_dir, "item_image.jpg")
                if not os.path.exists(image_path):
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H, W = image.shape[:2]

                # Load mask info
                mask_info_path = os.path.join(person_dir, "mask_info.json")
                with open(mask_info_path, 'r', encoding='utf-8') as f:
                    mask_info = json.load(f)

                # Combine masks (all classes)
                combined_mask = np.zeros((H, W), dtype=np.uint8)

                # Process each mask entry
                mask_entries = mask_info.get("item_masks", [])
                for mask_entry in mask_entries:
                    product_type = mask_entry.get("product_type", "unknown")

                    # 필터링 적용
                    if self.filter_type == 'tops' and product_type not in self.top_categories:
                        continue
                    elif self.filter_type == 'bottoms' and product_type not in self.bottom_categories:
                        continue

                    # 카테고리에 맞는 class_id 가져오기
                    if self.filter_type == 'tops':
                        class_id = self.top_categories.get(product_type, 0)
                    elif self.filter_type == 'bottoms':
                        class_id = self.bottom_categories.get(product_type, 0)
                    else:
                        class_id = 1  # 필터링 없으면 기본값 1

                    mask_path = mask_entry.get("path", None)
                    if mask_path and os.path.exists(mask_path):
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if binary_mask.shape != (H, W):
                            binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                        # Set class ID in the combined mask
                        combined_mask[binary_mask > 127] = class_id

                # Skip if no mask area
                if not np.any(combined_mask):
                    continue

                # Create colored mask visualization
                mask_colored = np.zeros((H, W, 4), dtype=np.float32)
                for class_id in range(1, max(len(self.top_categories), len(self.bottom_categories))):
                    mask = combined_mask == class_id
                    if np.any(mask):
                        color = self.colormap(class_id)
                        for c in range(3):
                            mask_colored[mask, c] = color[c]
                        mask_colored[mask, 3] = 0.5  # Alpha value (transparency)

                # Visualization
                plt.figure(figsize=(15, 10))

                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title(f'{self.filter_type.capitalize() if self.filter_type else "Item"} Original Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(combined_mask, cmap='tab20')
                plt.title(f'{self.filter_type.capitalize() if self.filter_type else "Item"} Mask (By Class)')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(image)
                plt.imshow(mask_colored)
                plt.title(f'{self.filter_type.capitalize() if self.filter_type else "Item"} Mask Overlay')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir,
                                         f'{self.filter_type if self.filter_type else "all"}_sample_{i}_visualization.png'))
                plt.close()

            except Exception as e:
                print(f"Visualization error ({person_dir}): {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Mask and Item Placement Analysis')
    parser.add_argument('--data-dir', type=str,
                        default="C:/Users/tjdwn/OneDrive/Desktop/parsingData/preprocessed/item",
                        help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--filter-type', type=str, default=None, choices=['tops', 'bottoms', None],
                        help='Filter type: tops or bottoms')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum number of samples to analyze')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable sample visualization')
    parser.add_argument('--analyze-all', default=True, action='store_true',
                        help='Analyze both tops and bottoms')

    args = parser.parse_args()

    if args.analyze_all:
        # 상의, 하의 모두 분석
        for filter_type in ['tops', 'bottoms']:
            print(f"\n=== Analyzing {filter_type} ===")
            analyzer = MaskAnalyzer(
                data_dir=args.data_dir,
                mode='item',  # item 모드만 사용
                output_dir=args.output_dir,
                filter_type=filter_type
            )

            analyzer.analyze_dataset(max_samples=args.max_samples)

            if args.visualize:
                analyzer.visualize_sample_masks(num_samples=5)
    else:
        # 지정된 필터 타입으로만 분석
        analyzer = MaskAnalyzer(
            data_dir=args.data_dir,
            mode='item',  # item 모드만 사용
            output_dir=args.output_dir,
            filter_type=args.filter_type
        )

        analyzer.analyze_dataset(max_samples=args.max_samples)

        if args.visualize:
            analyzer.visualize_sample_masks(num_samples=5)


if __name__ == '__main__':
    main()