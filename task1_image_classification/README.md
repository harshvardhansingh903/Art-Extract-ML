# Multi-Task Image Classification: ResNet50

Fine-grained classification of WikiArt images for **Style**, **Artist**, and **Genre** using a ResNet50 backbone with multi-task learning.

## Overview

This project implements a multi-task learning framework to simultaneously classify paintings across three dimensions:
- **Style**: Artistic style (e.g., impressionism, cubism)
- **Artist**: Painter/creator
- **Genre**: Subject matter (e.g., portrait, landscape)

### Architecture

```
Input Image (224×224)
    ↓
ResNet50 Backbone (pretrained ImageNet)
    ↓
Shared Features (2048-dim)
    ↓
┌─────────────┬──────────────┬─────────────┐
Style Head  Artist Head   Genre Head
    ↓             ↓             ↓
Style Logits  Artist Logits  Genre Logits
```

## Project Structure

```
task1_image_classification/
├── config.py                 # Configuration (learning rates, batch sizes, paths)
├── model.py                  # ResNet50 multi-task architecture
├── dataset.py                # WikiArt dataset loader and transforms
├── train.py                  # Training script with early stopping
├── evaluate.py               # Evaluation metrics (accuracy, precision, F1)
├── outlier_detection.py      # Low-confidence prediction detection
├── inference.py              # Single image prediction interface
├── utils.py                  # Helper functions (checkpoints, logging, plots)
├── checkpoints/              # Saved model checkpoints
├── results/                  # Metrics, plots, predictions
└── logs/                     # Training logs
```

## Setup & Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision, numpy, pandas, scikit-learn

### Installation

```bash
pip install torch torchvision torch-audio  # Adjust for your CUDA version
pip install numpy pandas scikit-learn pillow matplotlib seaborn
```

## Configuration

Edit `config.py` to customize:

```python
DATASET_CONFIG = {
    "root_dir": "path/to/images",
    "csv_file": "path/to/metadata.csv",
    "image_size": 224,
    "batch_size": 32,
    ...
}

TRAINING_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
    ...
}
```

## Usage

### 1. Training

```bash
python train.py
```

This will:
- Load dataset
- Initialize ResNet50 multi-task model
- Train with multi-task loss for multiple epochs
- Save best checkpoint based on validation loss
- Log metrics to `results/metrics.json`

### 2. Evaluation

```bash
python evaluate.py
```

Generates:
- Accuracy, precision, recall, F1-score per task
- Confusion matrices
- Evaluation metrics JSON

### 3. Outlier Detection

```bash
python outlier_detection.py
```

Identifies low-confidence predictions (possible misclassifications):
- Confidence score statistics
- Outlier samples below threshold
- Results saved to JSON

### 4. Inference

```python
from inference import ImageClassifier
from config import PATHS_CONFIG

classifier = ImageClassifier(
    checkpoint_path="checkpoints/best_model_epoch10.pth",
    label_info={'num_styles': 25, 'num_artists': 150, 'num_genres': 10}
)

predictions = classifier.predict("path/to/image.jpg")
print(predictions)
# Output:
# {
#     'style': {'class_idx': 5, 'confidence': 0.92},
#     'artist': {'class_idx': 42, 'confidence': 0.87},
#     'genre': {'class_idx': 3, 'confidence': 0.95}
# }
```

## Data Format

### CSV Metadata

Required columns:
- `image_name` or `image`: Image filename
- `style`: Style label
- `artist`: Artist label
- `genre`: Genre label
- `split` (optional): 'train' or 'val' (auto-split if missing)

Example:
```csv
image_name,style,artist,genre,split
painting_001.jpg,Impressionism,Claude Monet,Landscape,train
painting_002.jpg,Cubism,Pablo Picasso,Abstract,val
```

## Key Features

✅ **Multi-task Learning**: Single model learns 3 related tasks simultaneously  
✅ **Efficient Architecture**: ~47M parameters, ResNet50 backbone  
✅ **Robust Dataset Handling**: Recursive image discovery, flexible metadata matching  
✅ **Early Stopping**: Prevents overfitting on validation loss  
✅ **Comprehensive Evaluation**: Metrics, confusion matrices, confidence analysis  
✅ **Outlier Detection**: Identifies uncertain predictions  
✅ **Clean Inference**: Simple API for single/batch predictions  

## Training Tips

- **Class Imbalance**: Adjust loss weights in `LOSS_CONFIG` if needed
- **Large Dataset**: Use higher `batch_size` and `num_workers` on GPU
- **Limited Memory**: Reduce `batch_size` or freeze backbone layers
- **Transfer Learning**: `freeze_backbone=True` in `MODEL_CONFIG` for fine-tuning

## Performance Metrics

The model is evaluated on:
- **Per-task metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion matrices**: Shows classification errors per task
- **Confidence distribution**: Identifies model uncertainty

Example output:
```
STYLE: Acc=0.8234, F1=0.8156
ARTIST: Acc=0.7842, F1=0.7634
GENRE: Acc=0.8956, F1=0.8834
```

## Output Directory Structure

```
results/
├── metrics.json              # Training history
├── eval_metrics.json         # Evaluation scores
├── outliers.json            # Low-confidence samples
└── plots/
    ├── training_history.png # Loss & accuracy curves
    └── confusion_matrices.png
```

## Troubleshooting

**"No images found"**
- Verify image paths in CSV match filesystem
- Check dataset directory has proper structure

**"CUDA out of memory"**
- Reduce batch_size in config.py
- Enable gradient checkpointing

**Low validation accuracy**
- Check metadata labels for errors
- Increase training epochs
- Tune learning rate

## Code Quality

- **Modular design**: Each module handles single responsibility
- **Type hints**: Full type annotations for clarity
- **Minimal comments**: Code is self-documenting
- **DRY principle**: No redundant logic
- **~900 lines total**: Clean, production-ready codebase

## References

- [ResNet50 ImageNet Pretrained](https://torchvision.readthedocs.io/en/stable/models.html)
- [Multi-task Learning](https://arxiv.org/abs/1509.02595)
- [WikiArt Dataset](https://www.wikiart.org/)

---

**Last Updated**: 2024  
**License**: MIT
