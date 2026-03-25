"""
Simple evaluation metrics for similarity search.
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def compute_nearest_neighbors(
    embeddings: np.ndarray,
    metric: str = "cosine",
    top_k: int = 10,
) -> List[List[int]]:
    """
    Find top-k nearest neighbors for each embedding.
    
    Args:
        embeddings: Embedding matrix [N, D] (should be normalized)
        metric: 'cosine' or 'euclidean'
        top_k: Number of neighbors
    
    Returns:
        List of neighbor indices for each image
    """
    n_samples = embeddings.shape[0]
    
    if metric == "cosine":
        # For L2-normalized embeddings, cosine similarity = dot product
        similarities = np.dot(embeddings, embeddings.T)
        neighbors = [np.argsort(sim)[::-1][1:top_k+1] for sim in similarities]
    elif metric == "euclidean":
        distances = euclidean_distances(embeddings)
        neighbors = [np.argsort(dist)[1:top_k+1] for dist in distances]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return neighbors


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of embedding space.
    
    Args:
        embeddings: Embeddings [N, D]
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "mean_value": float(np.mean(embeddings)),
        "std_value": float(np.std(embeddings)),
    }
    return stats


def print_statistics(embeddings: np.ndarray) -> None:
    """
    Print embedding statistics.
    
    Args:
        embeddings: Embeddings [N, D]
    """
    stats = compute_embedding_statistics(embeddings)
    print("\n=== Embedding Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
