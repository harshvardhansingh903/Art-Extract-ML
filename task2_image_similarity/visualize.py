"""
Visualize image similarity results.

Displays query image and top K similar images with similarity scores.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
from typing import List, Tuple
from pathlib import Path


def visualize_similarity(
    query_image_path: str,
    similar_images: List[Tuple[str, float]],
    top_k: int = 5,
    figsize: Tuple[int, int] = (16, 4),
    title: str = "Image Similarity Results"
) -> None:
    """
    Visualize query image and top K similar images side by side.
    
    Args:
        query_image_path: Path to query image
        similar_images: List of (image_path, similarity_score) tuples
        top_k: Number of similar images to display
        figsize: Figure size (width, height)
        title: Figure title
    """
    # Take only top_k
    similar_images = similar_images[:top_k]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, top_k + 1, figsize=figsize, dpi=100)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Load and display query image
    try:
        query_img = Image.open(query_image_path).convert('RGB')
        axes[0].imshow(query_img)
        axes[0].set_title('Query Image', fontsize=11, fontweight='bold', pad=8)
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Error loading query:\n{str(e)[:30]}',
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # Load and display similar images
    for idx, (image_path, similarity) in enumerate(similar_images):
        ax = axes[idx + 1]
        
        try:
            img = Image.open(image_path).convert('RGB')
            ax.imshow(img)
            
            # Title with similarity score
            title_text = f'Top {idx + 1}\n{similarity:.4f}'
            ax.set_title(title_text, fontsize=10, fontweight='bold', pad=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=9)
        
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.05)
    
    return fig


def create_gallery(
    query_image_path: str,
    similar_images: List[Tuple[str, float]],
    top_k: int = 5,
    output_path: str = None
) -> None:
    """
    Create and save similarity visualization.
    
    Args:
        query_image_path: Path to query image
        similar_images: List of (image_path, similarity_score) tuples
        top_k: Number of similar images to display
        output_path: Path to save figure (optional)
    """
    fig = visualize_similarity(query_image_path, similar_images, top_k)
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"✓ Visualization saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    """Demo: visualize similarity results from inference.py"""
    
    from inference import search_similarity
    from config import PATHS_CONFIG, SIMILARITY_CONFIG
    import os
    
    print("="*60)
    print("Image Similarity: Visualization")
    print("="*60)
    print()
    
    # Paths
    embeddings_path = os.path.join(PATHS_CONFIG["results_dir"], "embeddings.npy")
    image_paths_file = os.path.join(PATHS_CONFIG["results_dir"], "image_paths.txt")
    
    # Check if embeddings exist
    if not os.path.exists(embeddings_path):
        print("✗ Embeddings not found!")
        print(f"  Please run: python train.py")
        exit(1)
    
    # Find sample query image
    images_dir = Path(__file__).parent / "data" / "nga_images"
    sample_images = sorted([
        str(f) 
        for f in images_dir.glob("*")
        if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.gif', '.bmp'}
    ])[:5]
    
    if sample_images:
        # Use first image as query
        query_image = sample_images[0]
        print(f"Query image: {Path(query_image).name}")
        print()
        
        # Search for similar images
        results = search_similarity(
            query_image_path=query_image,
            embeddings_path=embeddings_path,
            image_paths_file=image_paths_file,
            top_k=SIMILARITY_CONFIG["top_k"],
            device="cpu"
        )
        
        # Display results
        print("Displaying top 5 similar images...")
        create_gallery(
            query_image_path=query_image,
            similar_images=results,
            top_k=5,
            output_path="similarity_results.png"
        )
        
    else:
        print("✗ No images found in dataset!")
        exit(1)
