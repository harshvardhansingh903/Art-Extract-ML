# Model Details

## ResNet50 Architecture

Both tasks use ResNet50 as the backbone architecture.

### Base Architecture

- Input: 3-channel images (224×224)
- Backbone: ResNet50 (pre-trained on ImageNet)
- Output: Feature representations

### Task 1: Classification Heads

- Global Average Pooling on backbone output
- Separate dense layers for each classification task:
  - Style classification head
  - Artist classification head
  - Genre classification head

### Task 2: Embedding Layer

- Global Average Pooling on backbone output
- Dense embedding layer (typically 256 or 512 dimensions)
- L2 normalization for cosine similarity

## Training Details

### Task 1

- Loss: Cross-entropy loss
- Optimizer: Adam or SGD
- Learning rate: Configurable
- Data augmentation: Random crop, flip, color jitter

### Task 2

- Loss: Triplet loss or Contrastive loss (optional)
- Optimizer: Adam
- Learning rate: Configurable
- Data augmentation: Similar to Task 1

## Performance Metrics

### Task 1

- Accuracy per class
- F1-score
- Confusion matrices

### Task 2

- Cosine similarity distribution
- Recall@k
- mAP (mean Average Precision)
