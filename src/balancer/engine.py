import json
import os
import random
import cv2
import albumentations as A
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from .config import FolderBalanceConfig
from .utils import get_dataset_distribution, ensure_dir, list_image_files, process_file


class FolderImageDatasetBalancer:
    """
    Core engine for balancing and augmenting image classification datasets.

    The engine analyzes class distributions and applies undersampling (drop) or
    oversampling (augmentation) based on the specified configuration mode.
    """

    def __init__(self, config: FolderBalanceConfig):
        """
        Initializes the balancer with the provided configuration.

        Args:
            config: An instance of FolderBalanceConfig.
        """
        self.config = config
        random.seed(self.config.random_seed)

        self.aug_pipeline = self.config.augmentation_pipeline
        if self.aug_pipeline is None:
            self.aug_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
            ])

    def run(self) -> Dict[str, Any]:
        """
        Analyzes the dataset and performs the balancing operation.

        Returns:
            A dictionary containing a report of the operation results.
        """
        distribution = get_dataset_distribution(self.config.input_root)
        if not distribution:
            return {"status": "error", "message": "No classes found in input_root"}

        targets = self._calculate_targets(distribution)
        report = self._prepare_report(distribution, targets)

        if not self.config.dry_run:
            self._execute_balancing(distribution, targets)

        return report

    def _calculate_targets(self, distribution: Dict[str, int]) -> Dict[str, int]:
        """
        Determines the target number of images for each class based on mode.

        Args:
            distribution: The current distribution of class labels.

        Returns:
            A dictionary of class names to target image counts.
        """
        counts = list(distribution.values())
        min_c = min(counts)
        max_c = max(counts)
        avg_c = sum(counts) // len(counts)

        targets = {}
        for cls, count in distribution.items():
            if self.config.mode == "drop":
                target = min_c
            elif self.config.mode == "augment":
                target = max_c
            elif self.config.mode == "hybrid":
                target = avg_c
            else:
                target = count

            targets[cls] = target

        return targets

    def _prepare_report(self, dist: Dict[str, int], targets: Dict[str, int]) -> Dict[str, Any]:
        """
        Constructs a summary of the balancing plan.

        Args:
            dist: Current class distribution.
            targets: Target counts for each class.

        Returns:
            A dictionary report.
        """
        return {
            "status": "success",
            "config": {
                "mode": self.config.mode,
                "input": self.config.input_root,
                "output": self.config.output_root
            },
            "classes": {
                name: {"original": dist[name], "target": targets[name]}
                for name in dist
            }
        }

    def _execute_balancing(self, dist: Dict[str, int], targets: Dict[str, int]):
        """
        Physically processes the files based on original and target counts.

        Args:
            dist: Original distribution counts.
            targets: Target counts per class.
        """
        ensure_dir(self.config.output_root)

        for cls_name, original_count in dist.items():
            target_count = targets[cls_name]
            input_cls_path = os.path.join(self.config.input_root, cls_name)
            output_cls_path = os.path.join(self.config.output_root, cls_name)
            ensure_dir(output_cls_path)

            all_files = list_image_files(input_cls_path)
            random.shuffle(all_files)

            if target_count <= original_count:
                selected_files = all_files[:target_count]
                for filename in selected_files:
                    src = os.path.join(input_cls_path, filename)
                    dst = os.path.join(output_cls_path, filename)
                    process_file(src, dst, self.config.copy_instead_of_move)
            else:
                for filename in all_files:
                    src = os.path.join(input_cls_path, filename)
                    dst = os.path.join(output_cls_path, filename)
                    process_file(src, dst, self.config.copy_instead_of_move)

                remaining = target_count - original_count
                for i in range(remaining):
                    original_file = random.choice(all_files)
                    src = os.path.join(input_cls_path, original_file)
                    ext = Path(original_file).suffix
                    base = Path(original_file).stem
                    dst_name = f"{base}_aug_{i}{ext}"
                    dst = os.path.join(output_cls_path, dst_name)

                    image = cv2.imread(src)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        augmented = self.aug_pipeline(image=image)["image"]
                        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(dst, augmented)
                    else:
                        process_file(src, dst, self.config.copy_instead_of_move)
