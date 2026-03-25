"""
Configuration file for Task 2: Image Similarity System

Simple pretrained ResNet50 feature extraction approach.
"""

import os
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    "root_dir": str(PROJECT_ROOT / "task2_image_similarity" / "data" / "nga_images"),
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 0,  # Set to 0 for Windows compatibility
}

# ============================================================================
# MODEL CONFIGURATION (Pretrained ResNet50)
# ============================================================================

MODEL_CONFIG = {
    "pretrained": True,
    "embedding_dim": 2048,  # ResNet50 final feature dimension
}

# ============================================================================
# DEVICE & PATHS
# ============================================================================

DEVICE_CONFIG = {
    "device": "cpu",  # 'cuda' or 'cpu' - CPU is more accessible
}

PATHS_CONFIG = {
    "embeddings_dir": str(PROJECT_ROOT / "task2_image_similarity" / "results"),
    "results_dir": str(PROJECT_ROOT / "task2_image_similarity" / "results"),
}

# ============================================================================
# SIMILARITY SEARCH CONFIGURATION
# ============================================================================

SIMILARITY_CONFIG = {
    "metric": "cosine",  # 'cosine' or 'euclidean'
    "top_k": 10,  # Number of nearest neighbors to return
    "batch_size": 32,
}
