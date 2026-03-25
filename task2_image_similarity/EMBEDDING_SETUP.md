# Image Similarity: Embedding Generation & Search

This module generates embeddings for image similarity search using a pretrained ResNet50 feature extractor (no training required).

## Overview

- **Feature Extractor**: Pretrained ResNet50 (ImageNet weights)
- **Embedding Dimension**: 2048
- **Similarity Metric**: Cosine similarity (L2-normalized embeddings)
- **Dataset**: National Gallery of Art images (~2000 images)

## Setup

### 1. Ensure images are downloaded
```bash
cd image-ml-project
python download_nga_images.py
```
This downloads images to: `task2_image_similarity/data/nga_images/`

### 2. Configuration
Edit `config.py` to adjust:
- `batch_size`: Batch size for embedding generation (default: 32)
- `device`: "cpu" or "cuda" (auto-falls back to CPU if CUDA unavailable)
- `num_workers`: Set to 0 on Windows for compatibility

## Usage

### Step 1: Generate Embeddings
```bash
cd image-ml-project
python -m task2_image_similarity.train
```

**Output:**
- `task2_image_similarity/results/embeddings.npy` - Feature vectors [N, 2048]
- `task2_image_similarity/results/image_paths.txt` - Image file paths

**Time:** ~1.5-2 hours for 2000 images on CPU

### Step 2: Search for Similar Images
```bash
cd image-ml-project
python -m task2_image_similarity.inference
```

**Output:**
- Top 5 most similar images with cosine similarity scores
- Uses first image in dataset as query example

### Custom Query Search
```python
from task2_image_similarity.inference import search_similarity
import os

results = search_similarity(
    query_image_path="path/to/query/image.jpg",
    embeddings_path="task2_image_similarity/results/embeddings.npy",
    image_paths_file="task2_image_similarity/results/image_paths.txt",
    top_k=5,
    device="cpu"
)

for image_path, similarity in results:
    print(f"{image_path}: {similarity:.4f}")
```

## Architecture

### ResNet50 Feature Extractor
- Removes final classification layer
- Uses layer before average pooling for features
- Output: 2048-dimensional vectors
- L2-normalized for cosine similarity

### Similarity Computation
1. Normalize query embedding (L2)
2. Compute dot product with all gallery embeddings
3. Return top-K results by similarity score

## Performance

| Metric | Value |
|--------|-------|
| Feature Dimension | 2048 |
| Similarity Metric | Cosine |
| Time per Image | ~2.8s (CPU) |
| Memory per Image | ~8KB (float32) |
| Total Storage (2000 images) | ~16MB |

## File Structure
```
task2_image_similarity/
├── config.py                 # Configuration
├── model.py                  # ResNet50 feature extractor
├── dataset.py                # Image dataset loader
├── train.py                  # Embedding generation
├── inference.py              # Similarity search
├── data/
│   └── nga_images/           # Downloaded images
└── results/
    ├── embeddings.npy        # Generated embeddings
    └── image_paths.txt       # Image paths
```

## Notes

- **No Training**: Uses pretrained ImageNet weights
- **CPU Compatible**: Runs on CPU (slower but accessible)
- **Batch Processing**: Efficient batch-based embedding generation
- **Memory Efficient**: Processes images in batches, not all at once
