"""
Image dataset for similarity search.

Loads raw images for embedding generation (no labels needed).
"""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class ImageDataset(Dataset):
    """
    Simple image dataset for feature extraction.
    
    Loads images without labels for embedding generation.
    """
    
    def __init__(self, root_dir: str, image_size: int = 224):
        """
        Initialize dataset.
        
        Args:
            root_dir: Directory containing images
            image_size: Target image size
        """
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = self._find_images()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _find_images(self) -> List[str]:
        """Recursively find all image files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        images = []
        
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    images.append(os.path.join(root, file))
        
        return sorted(images)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get image and its path.
        
        Returns:
            Tuple of (image_tensor, image_path)
        """
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return black image on error
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return image, image_path


def create_dataloader(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> DataLoader:
    """
    Create dataloader for images.
    
    Args:
        root_dir: Directory with images
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
    
    Returns:
        DataLoader instance
    """
    dataset = ImageDataset(root_dir, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Important: maintain order for indexing
        pin_memory=True,
    )
