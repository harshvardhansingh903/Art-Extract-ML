# Image ML Project

A professional machine learning project for image classification and image similarity tasks using ResNet50.

## Project Overview

This project contains two main machine learning tasks:

1. **Task 1: Image Classification** - Classify images by Style, Artist, and Genre using CNN (ResNet50)
2. **Task 2: Image Similarity** - Find similar images using embedding-based approach with ResNet50 + Cosine Similarity

## Project Structure

```
image-ml-project/
├── task1_image_classification/     # Image Classification Task
├── task2_image_similarity/         # Image Similarity Task
├── shared_utils/                   # Shared utilities for both tasks
├── notebooks/                      # EDA and analysis notebooks
└── docs/                           # Documentation
```

## Getting Started

### Requirements
- Python 3.8+
- PyTorch
- See `requirements.txt` for full dependencies

### Installation

```bash
pip install -r requirements.txt
```

### Task 1: Image Classification

See [task1_image_classification/README.md](task1_image_classification/README.md)

### Task 2: Image Similarity

See [task2_image_similarity/README.md](task2_image_similarity/README.md)

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Model Details](docs/MODEL_DETAILS.md)

## License

MIT
