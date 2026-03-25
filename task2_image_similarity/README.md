# Task 2: Image Similarity System

Simple image similarity search using a pretrained ResNet50 feature extractor and cosine similarity matching.

## Overview

This system finds visually similar images using:
- **Feature Extractor**: Pretrained ResNet50 (frozen, no training)
- **Embeddings**: 2048-dimensional L2-normalized features
- **Search**: Cosine similarity in embedding space

## Architecture

```
Input Image (224×224)
    ↓
ResNet50 Backbone (Pretrained, Frozen)
    ↓
Flatten to 2048-dim features
    ↓
L2 Normalization
    ↓
Similarity Search (Cosine distance)
```

## Project Structure

```
task2_image_similarity/
├── config.py              # Configuration constants
├── dataset.py             # Image loading
├── model.py               # FeatureExtractor (ResNet50)
├── train.py               # Embedding generation
├── evaluate.py            # Statistics and metrics
├── inference.py           # Similarity search
├── utils.py               # Utility functions
├── README.md              # This file
├── embeddings/            # Precomputed embeddings (embeddings.npy + image_paths.txt)
├── results/               # Search results
└── checkpoints/           # (not used - model is pretrained)
```

## Setup

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- PIL

### Installation
```bash
pip install -r ../requirements.txt
```

## Usage

### Step 1: Generate Embeddings

Extract and save embeddings for all images once:

```python
from train import generate_embeddings
from config import DATASET_CONFIG, DEVICE_CONFIG

# Generate embeddings for image directory
embeddings, image_paths = generate_embeddings(
    root_dir="path/to/images",
    batch_size=32,
    device=DEVICE_CONFIG['device'],
    save_dir="./embeddings"
)
```

Output files:
- `embeddings/embeddings.npy`: Numpy array of shape [N, 2048]
- `embeddings/image_paths.txt`: Line-separated image paths

### Step 2: Search Similarity

Query images against the precomputed embeddings:

```python
from inference import search_similarity

# Search for similar images
results = search_similarity(
    query_image="path/to/query.jpg",
    embeddings_dir="./embeddings",
    top_k=10
)

# Results: list of (image_path, similarity_score) tuples
for image_path, score in results:
    print(f"{image_path}: {score:.4f}")
```

### Step 3: Evaluate

Compute statistics on the embedding space:

```python
from evaluate import print_statistics
import numpy as np

embeddings = np.load("./embeddings/embeddings.npy")
print_statistics(embeddings)
```

## Configuration

Edit `config.py` to modify:
- `DATASET_CONFIG`: Image size, batch size, file extensions
- `MODEL_CONFIG`: Use pretrained weights (always True)
- `SIMILARITY_CONFIG`: Search metric (cosine/euclidean), top_k results

## Key Features

- **No Training Required**: Uses pretrained ImageNet weights
- **Fast**: Simple feature extraction and cosine similarity
- **Scalable**: One-time embedding generation for large datasets
- **Normalized**: L2-normalized embeddings for accurate cosine distance

## How It Works

1. **Feature Extraction**: ResNet50 extracts 2048-dim features per image
2. **Normalization**: L2-normalize embeddings for cosine similarity
3. **Storage**: Save embeddings + image paths to disk (one-time)
4. **Search**: Query embedding vs. gallery embeddings using dot product (cosine distance on normalized vectors)

## Similarity Metric

Uses **Cosine Similarity** on L2-normalized embeddings:
- Similarity = dot_product(query_embedding, gallery_embedding)
- Range: [-1, 1] (typically [0, 1] for normalized vectors)
- Higher = more similar

## Expected Performance

- Embedding generation: ~50-100 images/second (GPU)
- Search query: <1ms per query
- Memory: ~8MB per 1000 images (2048-dim float32)

## Files

- **config.py**: Global constants
- **dataset.py**: ImageDataset class for loading images
- **model.py**: FeatureExtractor (ResNet50) + normalize_embeddings()
- **train.py**: generate_embeddings() function
- **inference.py**: SimilaritySearch class + search_similarity()
- **evaluate.py**: compute_nearest_neighbors(), compute_embedding_statistics()
- **utils.py**: Helper functions (load/save embeddings, device setup)

## Notes

- Model is pretrained on ImageNet for faster convergence
- Use GPU for training (13GB+ VRAM recommended)
- Embeddings are L2-normalized for cosine distance
- Supports large-scale retrieval with fast indexing

## Future Improvements

- [ ] Implement FAISS for large-scale indexing
- [ ] Add data augmentation strategies
- [ ] Support for multi-scale embeddings
- [ ] Implement hard negative mining
- [ ] Add batch hard triplet mining
