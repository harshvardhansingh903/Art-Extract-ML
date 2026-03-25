"""
Similarity search on precomputed embeddings.

Given query images, find most similar images from gallery.
"""

from typing import List, Tuple
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

from model import FeatureExtractor, normalize_embeddings
from config import DEVICE_CONFIG, PATHS_CONFIG, SIMILARITY_CONFIG


class SimilaritySearch:
    """
    Similarity search using precomputed embeddings.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        metric: str = "cosine",
        device: str = "cpu",
    ):
        """
        Initialize similarity search.
        
        Args:
            embeddings: Gallery embeddings [N, 2048]
            image_paths: Paths to gallery images
            metric: Similarity metric ('cosine' or 'euclidean')
            device: Device for computation
        """
        self.embeddings = embeddings  # Precomputed and normalized
        self.image_paths = image_paths
        self.metric = metric
        self.device = device
        
        # Load model for query embedding
        self.model = FeatureExtractor(pretrained=True).to(device)
        self.model.eval()
        
        # Transforms for query images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def search(
        self,
        query_image_path: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find similar images to query image.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
        
        Returns:
            List of (image_path, similarity_score) tuples
        """
        # Load and embed query
        query_embedding = self._embed_image(query_image_path)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.image_paths[idx], float(similarities[idx])))
        
        return results
    
    def _embed_image(self, image_path: str) -> np.ndarray:
        """Extract normalized embedding for image."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image)
            embedding = normalize_embeddings(embedding)
        
        return embedding.cpu().numpy()[0]


def search_similarity(
    query_image_path: str,
    embeddings_path: str,
    image_paths_file: str,
    top_k: int = 10,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Simple function to search for similar images.
    
    Args:
        query_image_path: Path to query image
        embeddings_path: Path to saved embeddings.npy
        image_paths_file: Path to saved image_paths.txt
        top_k: Number of results
        device: Device to use
    
    Returns:
        List of (image_path, similarity) tuples
    """
    # Load precomputed embeddings
    embeddings = np.load(embeddings_path)
    
    # Load image paths
    with open(image_paths_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    # Create searcher
    searcher = SimilaritySearch(embeddings, image_paths, device=device)
    
    # Search
    return searcher.search(query_image_path, top_k=top_k)


if __name__ == "__main__":
    """Run similarity search on a query image."""
    
    # Check if GPU is available, fall back to CPU
    device = DEVICE_CONFIG["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Paths
    embeddings_path = os.path.join(PATHS_CONFIG["results_dir"], "embeddings.npy")
    image_paths_file = os.path.join(PATHS_CONFIG["results_dir"], "image_paths.txt")
    
    # Check if embeddings exist
    if not os.path.exists(embeddings_path):
        print("✗ Embeddings not found!")
        print(f"  Please run: python train.py")
        exit(1)
    
    print("="*60)
    print("Image Similarity: Search Query")
    print("="*60)
    print(f"Device: {device}")
    print()
    
    # Try to find a sample query image from the dataset
    images_dir = str(Path(__file__).parent / "data" / "nga_images")
    sample_images = []
    
    if os.path.exists(images_dir):
        sample_images = [
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))
        ][:5]
    
    if sample_images:
        # Use first image as query
        query_image = sample_images[0]
        print(f"Query image: {query_image}")
        print()
        
        # Search
        results = search_similarity(
            query_image_path=query_image,
            embeddings_path=embeddings_path,
            image_paths_file=image_paths_file,
            top_k=SIMILARITY_CONFIG["top_k"],
            device=device,
        )
        
        print(f"Top {len(results)} Most Similar Images:")
        print("-" * 60)
        for rank, (image_path, similarity) in enumerate(results, 1):
            print(f"{rank}. {image_path:50s} | Similarity: {similarity:.4f}")
        
        print()
        print("="*60)
        print("✓ Similarity search complete!")
        print("="*60)
    else:
        print("✗ No images found in dataset directory!")
        print(f"  Expected: {images_dir}")
        exit(1)
