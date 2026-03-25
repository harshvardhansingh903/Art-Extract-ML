"""
Configuration file for Task 1: Image Classification
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    "root_dir": "D:/Gsoc/image-ml-project/data/wikiart",  # ✅ CORRECT - Full dataset
    "csv_file": "D:/Gsoc/image-ml-project/data/wikiart/metadata.csv",  # ✅ UPDATED
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "train_split": "train",
    "val_split": "val",
    "pin_memory": True,
}

# ============================================================================
# LABEL CONFIGURATION
# ============================================================================

# These will be automatically determined from the dataset,
# but you can override them here if needed

LABEL_CONFIG = {
    # "num_styles": 50,
    # "num_artists": 1000,
    # "num_genres": 25,
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "backbone": "resnet50",  # Backbone architecture
    "pretrained": True,  # Use ImageNet pre-trained weights
    "freeze_backbone": False,  # Fine-tune or train from scratch
    "dropout_rate": 0.5,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "adam",  # 'adam' or 'sgd'
    "scheduler": "cosine",  # 'cosine', 'step', or 'none'
    "gradient_clip": 1.0,  # Gradient clipping max norm
    "early_stopping_patience": 10,  # Epochs to wait before early stopping
}

# ============================================================================
# LOSS CONFIGURATION
# ============================================================================

LOSS_CONFIG = {
    "style_weight": 1.0,
    "artist_weight": 1.0,
    "genre_weight": 1.0,
    # Loss = w_style * loss_style + w_artist * loss_artist + w_genre * loss_genre
}

# ============================================================================
# DEVICE & PATHS
# ============================================================================

DEVICE_CONFIG = {
    "device": "cuda",  # 'cuda' or 'cpu'
    "mixed_precision": True,  # Use automatic mixed precision for faster training
}

PATHS_CONFIG = {
    "checkpoint_dir": "./checkpoints",
    "results_dir": "./results",
    "log_dir": "./logs",
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    "save_predictions": True,
    "save_plots": True,
    "compute_confusion_matrix": True,
}
