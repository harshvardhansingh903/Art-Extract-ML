"""
Generate final image similarity results for project submission.

Selects 3 diverse query images and creates visualizations with top 5 similar images.
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import List, Tuple
from pathlib import Path
import os


def load_embeddings_and_paths(results_dir: str):
    """Load precomputed embeddings and image paths."""
    embeddings_path = os.path.join(results_dir, "embeddings.npy")
    image_paths_file = os.path.join(results_dir, "image_paths.txt")
    
    embeddings = np.load(embeddings_path)
    with open(image_paths_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    return embeddings, image_paths


def find_similar_images(
    query_idx: int,
    embeddings: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """Find top K similar images to query using cosine similarity."""
    query_embedding = embeddings[query_idx]
    
    # Compute cosine similarities with all images
    similarities = np.dot(embeddings, query_embedding)
    
    # Get top K (excluding the query itself at index 0)
    top_indices = np.argsort(similarities)[::-1][:top_k + 1]
    
    # Filter out query image itself
    results = []
    for idx in top_indices:
        if idx != query_idx:
            results.append((idx, float(similarities[idx])))
            if len(results) == top_k:
                break
    
    return results


def select_diverse_queries(
    image_paths: List[str],
    num_images: int = 3
) -> List[Tuple[int, str]]:
    """
    Select diverse query images from dataset.
    Heuristic: select images spread across the dataset.
    """
    n = len(image_paths)
    indices = []
    
    # Select images uniformly spread across dataset
    step = n // (num_images + 1)
    for i in range(1, num_images + 1):
        indices.append(i * step)
    
    return [(idx, image_paths[idx]) for idx in indices]


def create_similarity_visualization(
    query_idx: int,
    query_image_path: str,
    similar_results: List[Tuple[int, float]],
    image_paths: List[str],
    figsize: Tuple[int, int] = (18, 4)
) -> Tuple[plt.Figure, dict]:
    """Create visualization for a query image and its similar matches."""
    
    fig, axes = plt.subplots(1, 6, figsize=figsize, dpi=100)
    
    # Load and display query image
    try:
        query_img = Image.open(query_image_path).convert('RGB')
        axes[0].imshow(query_img)
        axes[0].set_title('Query Image', fontsize=11, fontweight='bold', color='darkblue', pad=10)
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
        axes[0].axis('off')
    
    # Display similar images
    scores_data = []
    for i, (similar_idx, similarity) in enumerate(similar_results):
        ax = axes[i + 1]
        similar_path = image_paths[similar_idx]
        
        try:
            img = Image.open(similar_path).convert('RGB')
            ax.imshow(img)
            
            # Title with rank and similarity score
            title = f'Rank {i+1}\nScore: {similarity:.4f}'
            ax.set_title(title, fontsize=10, fontweight='bold', color='darkgreen', pad=10)
            
            scores_data.append({
                'rank': i + 1,
                'similarity': similarity,
                'image': Path(similar_path).name
            })
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', fontsize=9)
        
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    return fig, {'query_image': Path(query_image_path).name, 'results': scores_data}


def print_results_table(query_num: int, results_data: dict):
    """Print similarity results in a readable table format."""
    print()
    print("="*70)
    print(f"QUERY IMAGE {query_num}: {results_data['query_image']}")
    print("="*70)
    print(f"{'Rank':<6} {'Similarity Score':<20} {'Similar Image':<45}")
    print("-"*70)
    
    for result in results_data['results']:
        print(f"{result['rank']:<6} {result['similarity']:<20.6f} {result['image']:<45}")
    
    print("="*70)


def main():
    """Generate final results for project submission."""
    
    print("\n" + "="*70)
    print("IMAGE SIMILARITY SYSTEM - FINAL RESULTS GENERATION")
    print("="*70 + "\n")
    
    # Configuration
    results_dir = Path(__file__).parent / "results"
    images_dir = Path(__file__).parent / "data" / "nga_images"
    output_dir = Path(__file__).parent / "results"
    
    # Load precomputed embeddings
    print("Loading precomputed embeddings...")
    embeddings, image_paths = load_embeddings_and_paths(str(results_dir))
    print(f"✓ Loaded {len(image_paths)} embeddings of dimension {embeddings.shape[1]}\n")
    
    # Select diverse query images
    print("Selecting diverse query images from dataset...")
    query_images = select_diverse_queries(image_paths, num_images=3)
    print(f"✓ Selected {len(query_images)} query images\n")
    
    # Process each query image
    all_results = []
    for query_num, (query_idx, query_path) in enumerate(query_images, 1):
        print(f"Processing Query {query_num}...")
        print(f"  Image: {Path(query_path).name}")
        
        # Find similar images
        similar_results = find_similar_images(query_idx, embeddings, top_k=5)
        
        # Create visualization
        fig, results_data = create_similarity_visualization(
            query_idx=query_idx,
            query_image_path=query_path,
            similar_results=similar_results,
            image_paths=image_paths
        )
        
        # Save figure
        output_filename = f"similarity_{query_num}.png"
        output_path = output_dir / output_filename
        fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
        print(f"  ✓ Saved: {output_filename}")
        
        # Store results
        all_results.append((query_num, results_data))
        
        # Print results
        print_results_table(query_num, results_data)
        
        plt.close(fig)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    for i in range(1, 4):
        filename = f"similarity_{i}.png"
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename:<30} ({size_kb:.1f} KB)")
    
    print("\n✓ Ready for GitHub and report submission!\n")


if __name__ == "__main__":
    main()
