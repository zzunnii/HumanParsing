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

    def __init__(self, data_dir, mode='model', output_dir='analysis_results'):
        """
        Args:
            data_dir: Dataset path (parent directory containing train, val folders)
            mode: 'model' or 'item'
            output_dir: Path to save analysis results
        """
        self.data_dir = data_dir
        self.mode = mode
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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
        for person_dir in tqdm(person_dirs, desc="Analyzing Person Data"):
            self.analyze_person(person_dir)

        # Calculate and visualize overall statistics
        self.calculate_statistics()

    def analyze_person(self, person_dir):
        """Analyze a single person directory"""
        try:
            # Load image
            image_path = os.path.join(person_dir, f"{self.mode}_image.jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(person_dir, "model_image.jpg")

            if not os.path.exists(image_path):
                return  # Skip if image doesn't exist

            image = cv2.imread(image_path)
            if image is None:
                return

            H, W = image.shape[:2]
            total_image_area = H * W
            self.total_areas.append(total_image_area)

            # Load mask info
            mask_info_path = os.path.join(person_dir, "mask_info.json")
            if not os.path.exists(mask_info_path):
                return

            with open(mask_info_path, 'r', encoding='utf-8') as f:
                mask_info = json.load(f)

            # Combine masks (all classes)
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            any_mask = np.zeros((H, W), dtype=np.bool_)  # Binary mask (all classes)

            # Process each mask entry
            mask_entries = mask_info.get(f"{self.mode}_masks", [])
            for mask_entry in mask_entries:
                category = mask_entry.get("category", "unknown")

                # Try to extract class_id from category string if it's a digit
                try:
                    class_id = int(category) if category.isdigit() else 0
                except:
                    # If category is not a direct digit, try other parsing methods
                    if isinstance(category, str) and '_' in category:
                        parts = category.split('_')
                        if parts[-1].isdigit():
                            class_id = int(parts[-1])
                        else:
                            class_id = 0
                    else:
                        class_id = 0

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
                    self.class_pixel_counts[category] += pixel_count

            # Add to total pixel count
            self.total_pixels += H * W

            # Skip if no person area found
            if not np.any(any_mask):
                return

            # Calculate person area in pixels
            person_pixel_area = np.sum(any_mask)
            self.person_areas.append(person_pixel_area)

            # Calculate person area ratio
            person_ratio = person_pixel_area / total_image_area
            self.person_ratios.append(person_ratio)

            # Calculate person boundaries
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

        except Exception as e:
            print(f"Analysis error ({person_dir}): {str(e)}")

    def calculate_statistics(self):
        """Calculate and visualize statistics from collected data"""
        if not self.person_ratios:
            print("No data to analyze.")
            return

        # 1. Person ratio histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.person_ratios, bins=20, alpha=0.7)
        plt.title('Person Area Ratio Distribution')
        plt.xlabel('Person Ratio (0-1)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'person_ratio_histogram.png'))

        # 2. Person center point distribution
        plt.figure(figsize=(10, 8))
        centers = np.array(self.person_centers)
        plt.scatter(centers[:, 0], centers[:, 1], alpha=0.5, s=10)
        plt.title('Person Center Distribution')
        plt.xlabel('X Coordinate (0-1)')
        plt.ylabel('Y Coordinate (0-1)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.savefig(os.path.join(self.output_dir, 'person_center_distribution.png'))

        # 3. Person height/width ratio scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.person_widths, self.person_heights, alpha=0.5, s=10)
        plt.title('Person Size Ratio Distribution')
        plt.xlabel('Width Ratio (0-1)')
        plt.ylabel('Height Ratio (0-1)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'person_size_distribution.png'))

        # 4. Class pixel ratios
        class_ratios = {k: v / self.total_pixels for k, v in self.class_pixel_counts.items()}
        plt.figure(figsize=(12, 6))
        plt.bar(class_ratios.keys(), class_ratios.values())
        plt.title('Class Pixel Ratios')
        plt.xlabel('Class')
        plt.ylabel('Ratio (0-1)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_pixel_ratios.png'))

        # 5. Person area distribution (in pixels)
        plt.figure(figsize=(10, 6))
        plt.hist(self.person_areas, bins=30, alpha=0.7)
        plt.title('Person Area Distribution (in pixels)')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'person_area_distribution.png'))

        # 6. Statistics summary
        summary = {
            'person_ratio': {
                'mean': float(np.mean(self.person_ratios)),
                'median': float(np.median(self.person_ratios)),
                'min': float(np.min(self.person_ratios)),
                'max': float(np.max(self.person_ratios))
            },
            'person_height': {
                'mean': float(np.mean(self.person_heights)),
                'median': float(np.median(self.person_heights)),
                'min': float(np.min(self.person_heights)),
                'max': float(np.max(self.person_heights))
            },
            'person_width': {
                'mean': float(np.mean(self.person_widths)),
                'median': float(np.median(self.person_widths)),
                'min': float(np.min(self.person_widths)),
                'max': float(np.max(self.person_widths))
            },
            'person_area_pixels': {
                'mean': float(np.mean(self.person_areas)),
                'median': float(np.median(self.person_areas)),
                'min': float(np.min(self.person_areas)),
                'max': float(np.max(self.person_areas))
            },
            'person_center_x': {
                'mean': float(np.mean([c[0] for c in self.person_centers])),
                'median': float(np.median([c[0] for c in self.person_centers]))
            },
            'person_center_y': {
                'mean': float(np.mean([c[1] for c in self.person_centers])),
                'median': float(np.median([c[1] for c in self.person_centers]))
            },
            'class_ratios': class_ratios
        }

        # Save statistics
        with open(os.path.join(self.output_dir, 'statistics_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print("Statistics analysis complete!")
        print(
            f"Person ratio - Mean: {summary['person_ratio']['mean']:.4f}, Median: {summary['person_ratio']['median']:.4f}")
        print(
            f"Person height - Mean: {summary['person_height']['mean']:.4f}, Median: {summary['person_height']['median']:.4f}")
        print(
            f"Person width - Mean: {summary['person_width']['mean']:.4f}, Median: {summary['person_width']['median']:.4f}")
        print(
            f"Person center X - Mean: {summary['person_center_x']['mean']:.4f}, Median: {summary['person_center_x']['median']:.4f}")
        print(
            f"Person center Y - Mean: {summary['person_center_y']['mean']:.4f}, Median: {summary['person_center_y']['median']:.4f}")
        print(
            f"Person area (pixels) - Mean: {summary['person_area_pixels']['mean']:.1f}, Median: {summary['person_area_pixels']['median']:.1f}")

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

        # Select random samples
        import random
        random.shuffle(person_dirs)
        samples = person_dirs[:num_samples]

        for i, person_dir in enumerate(samples):
            try:
                # Load image
                image_path = os.path.join(person_dir, f"{self.mode}_image.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(person_dir, "model_image.jpg")

                if not os.path.exists(image_path):
                    continue  # Skip if image doesn't exist

                image = cv2.imread(image_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H, W = image.shape[:2]

                # Load mask info
                mask_info_path = os.path.join(person_dir, "mask_info.json")
                if not os.path.exists(mask_info_path):
                    continue

                with open(mask_info_path, 'r', encoding='utf-8') as f:
                    mask_info = json.load(f)

                # Combine masks (all classes)
                combined_mask = np.zeros((H, W), dtype=np.uint8)

                # Process each mask entry
                mask_entries = mask_info.get(f"{self.mode}_masks", [])
                for mask_entry in mask_entries:
                    category = mask_entry.get("category", "unknown")
                    class_id = int(category.split('_')[-1]) if isinstance(category, str) and '_' in category and \
                                                               category.split('_')[-1].isdigit() else 0

                    mask_path = mask_entry.get("path", None)
                    if mask_path and os.path.exists(mask_path):
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if binary_mask.shape != (H, W):
                            binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                        # Set class ID in the combined mask
                        combined_mask[binary_mask > 127] = class_id

                # Skip if no person area
                if not np.any(combined_mask):
                    continue

                # Create colored mask visualization
                mask_colored = np.zeros((H, W, 4), dtype=np.float32)
                for class_id in range(1, 21):  # Exclude background (0)
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
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(combined_mask, cmap='tab20')
                plt.title('Mask (By Class)')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(image)
                plt.imshow(mask_colored)
                plt.title('Mask Overlay')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'sample_{i}_visualization.png'))
                plt.close()

            except Exception as e:
                print(f"Visualization error ({person_dir}): {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Mask and Person Placement Analysis')
    parser.add_argument('--data-dir', type=str,
                        default="C:/Users/tjdwn/OneDrive/Desktop/parsingData/preprocessed/model",
                        help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--mode', type=str, default='model', choices=['model', 'item'],
                        help='Data mode to analyze')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum number of samples to analyze')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable sample visualization')

    args = parser.parse_args()

    analyzer = MaskAnalyzer(
        data_dir=args.data_dir,
        mode=args.mode,
        output_dir=args.output_dir
    )

    analyzer.analyze_dataset(max_samples=args.max_samples)

    if args.visualize:
        analyzer.visualize_sample_masks(num_samples=5)


if __name__ == '__main__':
    main()