# 🎨 ArtExtract-ML: Deep Learning for Artwork Classification & Similarity

> End-to-end deep learning system for artwork understanding: multi-task classification (style, artist, genre) and visual similarity search using large-scale art datasets.

---

## 🚀 Overview

This project implements a **comprehensive AI pipeline for artwork analysis**, developed as part of the *ArtExtract Evaluation Test*. It consists of two main components:

### 🔹 Task 1: Multi-Task Image Classification
- Predicts:
  - 🎭 **Style** (e.g., Impressionism, Cubism)
  - 🧑‍🎨 **Artist** (e.g., Van Gogh, Monet)
  - 🖼️ **Genre** (e.g., Portrait, Landscape)
- Uses a **ResNet50-based multi-head architecture**

### 🔹 Task 2: Image Similarity Search
- Finds visually similar paintings
- Uses **deep feature embeddings + cosine similarity**
- Enables content-based image retrieval

---

## 🧠 Model Architecture

### Task 1: Classification Model
- Backbone: **ResNet50 (pretrained)**
- Multi-task heads:
  - Style classifier
  - Artist classifier
  - Genre classifier
- Loss: Combined Cross-Entropy Loss

> 📌 Note: A Convolutional-Recurrent (CRNN) architecture was considered, but CNN alone is sufficient since paintings are static images without sequential dependencies.

---

### Task 2: Similarity Model
- Feature extractor: **ResNet50 (frozen)**
- Output: **2048-dimensional embeddings**
- Similarity metric: **Cosine similarity**

---

## 📊 Datasets Used

### 🎨 WikiArt Dataset
- Source: ArtGAN WikiArt dataset
- ~80,000+ images
- Used for classification (Task 1)
- https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md

### 🏛️ National Gallery of Art Dataset
- Source: NGA Open Data
- ~2,000 images (sampled)
- Used for similarity search (Task 2)
- https://github.com/NationalGalleryOfArt/opendata

> ⚠️ Datasets are not included due to size constraints.

---

## 📈 Results

### 🔹 Task 1 Performance

| Task   | Accuracy | Precision | Recall | F1 Score |
|--------|---------|----------|--------|---------|
| Style  | 0.82    | 0.81     | 0.82   | 0.81    |
| Artist | 0.71    | 0.70     | 0.71   | 0.71    |
| Genre  | 0.85    | 0.85     | 0.85   | 0.85    |

### 📊 Additional Outputs
- Confusion Matrices
- Sample Predictions
- Outlier Detection (low-confidence predictions)

---

### 🔹 Task 2 Results

- Embeddings generated: **1944 × 2048**
- Similarity retrieval using cosine similarity
- Visual results generated for top-K matches

---

## 🖼️ Sample Outputs

### 🔹 Similarity Search
- Query image + Top 5 similar artworks
- Cosine similarity scores displayed

### 🔹 Classification
- Predicted labels with confidence scores
- Misclassified examples analyzed

---

## 📁 Project Structure

Gsoc\image-ml-project/
│
├── 📄 README.md                          # Main project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 download_nga_images.py             # Script to download National Gallery of Art images
│
├── 📁 data/                              # Datasets
│   ├── 📁 wikiart/                       # WikiArt images (1000s of artwork files)
│   │   ├── artist images for training/validation
│   │   │   (Pavel, Vladimir, Wassily, etc.)
│   │   └── ...
│   └── 📁 wikiart_csv/                   # Metadata for WikiArt dataset
│       ├── artist_class.txt              # Artist class mappings
│       ├── artist_train.csv              # Training splits for artists
│       ├── artist_val.csv                # Validation splits for artists
│       ├── genre_class.txt               # Genre class mappings
│       ├── genre_train.csv               # Training splits for genres
│       ├── genre_val.csv                 # Validation splits for genres
│       ├── style_class.txt               # Style class mappings
│       ├── style_train.csv               # Training splits for styles
│       └── style_val.csv                 # Validation splits for styles
│
├── 📁 shared_utils/                      # Shared utilities
│   ├── __init__.py
│   └── common_utils.py                   # Common utility functions
│
├── 📁 docs/                              # Documentation
│   ├── ARCHITECTURE.md                   # System architecture details
│   └── MODEL_DETAILS.md                  # Model specifications
│
├── 📁 notebooks/                         # Jupyter notebooks (empty)
│
├── 📁 logs/                              # Training logs
│
├── 📁 results/                           # Project outputs
│
└── 📁 task1_image_classification/        # Task 1: Multi-Task Classification
    │
    ├── 📄 README.md                      # Task 1 documentation
    ├── 📄 TECHNICAL_REPORT.md            # 🆕 Comprehensive technical report ⭐
    ├── 📄 REFACTORING_SUMMARY.md         # Code refactoring notes
    │
    ├── 📄 config.py                      # Configuration for model/training
    ├── 📄 model.py                       # ResNet50MultiTask architecture
    ├── 📄 dataset.py                     # WikiArtDataset loader
    ├── 📄 train.py                       # Training script
    ├── 📄 resume_training.py             # Resume training from checkpoint
    ├── 📄 evaluate.py                    # Evaluation metrics
    ├── 📄 inference.py                   # Inference pipeline
    ├── 📄 outlier_detection.py           # Low-confidence detection
    ├── 📄 utils.py                       # Utility functions
    ├── 📄 generate_results.py            # Initial results generator
    ├── 📄 generate_results_simple.py     # Simplified results generator
    │
    ├── 📁 data/                          # Dataset storage
    │   └── 📁 wikiart/
    │       ├── metadata.csv
    │       └── 📁 images/                # WikiArt images
    │
    ├── 📁 checkpoints/                   # Trained model checkpoints
    │   ├── best_model_epoch0.pth
    │   ├── best_model_epoch1_epoch1.pth
    │   ├── best_model_epoch2_epoch2.pth
    │   ├── best_model_epoch3.pth
    │   └── best_model_epoch4.pth
    │
    ├── 📁 logs/                          # Training logs
    │
    ├── 📁 results/                       # Evaluation results
    │   ├── 📄 metrics.json               # Performance metrics (0.6 KB)
    │   ├── 📁 plots/
    │   │   ├── confusion_matrices.png    # 3 confusion matrices (38.8 KB)
    │   │   └── metrics_comparison.png    # Comparison charts (74 KB)
    │   ├── 📁 predictions/
    │   │   └── predictions.txt           # Sample predictions (0.9 KB)
    │   ├── 📁 outliers/
    │   │   └── outliers.txt              # Misclassified samples (0.3 KB)
    │   └── 📁 embeddings/                # Feature embeddings
    │
    └── 📁 __pycache__/                   # Compiled Python files
│
└── 📁 task2_image_similarity/            # Task 2: Image Similarity Search
    │
    ├── 📄 README.md                      # Task 2 documentation
    ├── 📄 EMBEDDING_SETUP.md             # Embedding generation details
    │
    ├── 📄 config.py                      # Configuration for similarity
    ├── 📄 model.py                       # Feature extraction model
    ├── 📄 dataset.py                     # Image dataset loader
    ├── 📄 train.py                       # Generate embeddings
    ├── 📄 inference.py                   # Similarity search
    ├── 📄 evaluate.py                    # Evaluation metrics
    ├── 📄 visualize.py                   # Visualization utilities
    ├── 📄 utils.py                       # Common utilities
    ├── 📄 generate_final_results.py      # Generate result images
    ├── 📄 similarity_results.png         # Sample similarity output
    │
    ├── 📁 data/                          # Image storage
    │   └── 📁 nga_images/                # 1944 NGA images (downloaded)
    │
    ├── 📁 checkpoints/                   # Model checkpoints
    │
    ├── 📁 embeddings/                    # Embedding cache
    │
    ├── 📁 logs/                          # Training logs
    │
    ├── 📁 results/                       # Results and outputs
    │   ├── 📄 embeddings.npy             # 1944×2048 feature vectors (16 MB)
    │   ├── 📄 image_paths.txt            # Paths to images
    │   ├── 📄 metrics.json               # Performance metrics
    │   ├── 📁 embeddings/                # Individual embeddings
    │   ├── 📁 similarity_results/        # Similarity search outputs
    │   ├── 📄 similarity_1.png           # Result visualization 1 (700.8 KB)
    │   └── 📄 similarity_3.png           # Result visualization 3 (1465.6 KB)
    │
    └── 📁 __pycache__/                   # Compiled Python files

