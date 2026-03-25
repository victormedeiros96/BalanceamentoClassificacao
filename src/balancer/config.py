from dataclasses import dataclass
from typing import Literal, Any, Optional


@dataclass
class FolderBalanceConfig:
    """
    Configuration for the FolderImageDatasetBalancer.

    Attributes:
        input_root: Path to the source dataset directory.
        output_root: Path to the destination balanced dataset directory.
        mode: Balancing strategy ("drop", "augment", or "hybrid").
        final_tolerance: The allowed variance between classes in the final result.
        intermediate_tolerance: Threshold used specifically for the hybrid mode.
        random_seed: Seed for reproducibility in random sampling.
        copy_instead_of_move: Whether to duplicate or move existing files.
        dry_run: If True, only calculations are done without modifying the disk.
        augmentation_pipeline: An explicit Albumentations pipeline. If None, a default one provides geometric and color transformations.
    """
    input_root: str
    output_root: str
    mode: Literal["drop", "augment", "hybrid"] = "hybrid"
    final_tolerance: float = 0.10
    intermediate_tolerance: float = 0.30
    random_seed: int = 42
    copy_instead_of_move: bool = True
    dry_run: bool = False
    augmentation_pipeline: Optional[Any] = None
