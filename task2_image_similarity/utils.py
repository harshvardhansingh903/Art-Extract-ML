"""
Utility functions for image similarity task.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get computation device.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def ensure_dir_exists(dir_path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Directory path to create
    
    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_embeddings(embeddings_path: str, image_paths_file: str) -> Tuple[np.ndarray, list]:
    """
    Load precomputed embeddings and image paths.
    
    Args:
        embeddings_path: Path to embeddings.npy file
        image_paths_file: Path to image_paths.txt file
    
    Returns:
        Tuple of (embeddings array, list of image paths)
    """
    embeddings = np.load(embeddings_path)
    with open(image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    return embeddings, image_paths


def save_embeddings(
    embeddings: np.ndarray,
    image_paths: list,
    save_dir: str
) -> None:
    """
    Save embeddings and image paths to disk.
    
    Args:
        embeddings: Embedding array [N, D]
        image_paths: List of image paths
        save_dir: Directory to save files
    """
    ensure_dir_exists(save_dir)
    
    # Save embeddings
    embeddings_path = os.path.join(save_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    
    # Save image paths
    paths_file = os.path.join(save_dir, "image_paths.txt")
    with open(paths_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')


def normalize_embeddings(embeddings: np.ndarray, p: float = 2) -> np.ndarray:
    """
    Normalize embeddings using Lp norm.
    
    Args:
        embeddings: Input embeddings [N, D]
        p: Norm order (2 for L2 norm)
    
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, ord=p, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def dict_to_string(d: dict, indent: int = 0) -> str:
    """
    Convert dictionary to formatted string.
    
    Args:
        d: Dictionary to convert
        indent: Indentation level
    
    Returns:
        Formatted string representation
    """
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append("  " * indent + f"{key}:")
            lines.append(dict_to_string(value, indent + 1))
        else:
            lines.append("  " * indent + f"{key}: {value}")
    return "\n".join(lines)
