import os
import random
import shutil
from pathlib import Path
from typing import Dict, List


def list_image_files(directory: str) -> List[str]:
    """
    Lists all image files in a given directory based on common extensions.

    Args:
        directory: The path to search.

    Returns:
        A list of relative paths from the directory.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = []
    for f in os.listdir(directory):
        if Path(f).suffix.lower() in extensions:
            files.append(f)
    return files


def get_dataset_distribution(root_dir: str) -> Dict[str, int]:
    """
    Counts the number of image files in each class subfolder.

    Args:
        root_dir: The root path of the dataset.

    Returns:
        A dictionary where keys are class names and values are image counts.
    """
    distribution = {}
    path = Path(root_dir)
    if not path.exists():
        return distribution

    for item in path.iterdir():
        if item.is_dir():
            files = list_image_files(str(item))
            distribution[item.name] = len(files)
    return distribution


def ensure_dir(directory: str):
    """
    Creates a directory if it does not already exist.

    Args:
        directory: The path to ensure exists.
    """
    os.makedirs(directory, exist_ok=True)


def process_file(src: str, dst: str, copy: bool):
    """
    Copies or moves a file from source to destination.

    Args:
        src: Source file path.
        dst: Destination file path.
        copy: If True, copies the file; otherwise, moves it.
    """
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)
