"""
Generate embeddings for all images in the dataset.

No training - just feature extraction and storage.
"""

from typing import Tuple, List
import torch
import numpy as np
from pathlib import Path
import os

from model import FeatureExtractor, normalize_embeddings
from dataset import create_dataloader
from config import DATASET_CONFIG, DEVICE_CONFIG, PATHS_CONFIG


def generate_embeddings(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cpu",
    save_dir: str = "./results",
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate embeddings for all images in a directory.
    
    Args:
        root_dir: Directory containing images
        batch_size: Batch size for processing
        num_workers: Number of workers
        device: Device to use
        save_dir: Directory to save embeddings
    
    Returns:
        Tuple of (embeddings_array, image_paths_list)
    """
    # Create dataloader
    dataloader = create_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Load model
    model = FeatureExtractor(pretrained=True).to(device)
    model.eval()
    
    embeddings_list = []
    image_paths_list = []
    
    print(f"Generating embeddings for {len(dataloader.dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            # L2 normalize
            features = normalize_embeddings(features)
            
            # Collect
            embeddings_list.append(features.cpu().numpy())
            image_paths_list.extend(paths)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} / {len(dataloader.dataset)}")
    
    # Stack all embeddings
    embeddings = np.vstack(embeddings_list)
    
    # Save embeddings
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    
    # Save image paths
    with open(os.path.join(save_dir, "image_paths.txt"), "w") as f:
        for path in image_paths_list:
            f.write(path + "\n")
    
    print(f"✓ Embeddings saved: {embeddings.shape}")
    print(f"✓ Image paths saved: {len(image_paths_list)} images")
    
    return embeddings, image_paths_list


if __name__ == "__main__":
    """Generate embeddings for all images in the dataset."""
    
    # Check if GPU is available, fall back to CPU
    device = DEVICE_CONFIG["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("="*60)
    print("Image Similarity: Embedding Generation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image directory: {DATASET_CONFIG['root_dir']}")
    print(f"Results directory: {PATHS_CONFIG['results_dir']}")
    print()
    
    # Generate embeddings
    embeddings, image_paths = generate_embeddings(
        root_dir=DATASET_CONFIG["root_dir"],
        batch_size=DATASET_CONFIG["batch_size"],
        num_workers=DATASET_CONFIG["num_workers"],
        device=device,
        save_dir=PATHS_CONFIG["results_dir"],
    )
    
    print()
    print("="*60)
    print("✓ Embedding generation complete!")
    print("="*60)
