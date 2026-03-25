"""Utility functions for training and evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def setup_directories(config_paths: Dict[str, str]) -> None:
    """Create necessary directories."""
    for path in config_paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device(device_name: str = "cuda") -> torch.device:
    """Get torch device with fallback to CPU."""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                   metrics: Dict, checkpoint_dir: str, name: str = "best_model") -> str:
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f"{name}_epoch{epoch}.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, checkpoint_path)
    return str(checkpoint_path)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str,
                   optimizer: torch.optim.Optimizer = None, device: str = "cuda") -> Tuple[int, Dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def save_metrics_json(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    with open(output_path, "w") as f:
        json.dump(metrics_serializable, f, indent=4)


def format_metrics_for_logging(metrics: Dict, stage: str = "val") -> str:
    """Format metrics for console output."""
    lines = [f"{stage}: Loss={metrics['loss']:.4f}"]
    for task in ['style', 'artist', 'genre']:
        if f'acc_{task}' in metrics:
            lines.append(f"  {task}={metrics[f'acc_{task}']:.4f}")
    return " | ".join(lines)


def plot_training_history(history: Dict, output_dir: str = "./results/plots") -> None:
    """Plot training history."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    train_accs = [history.get(f'train_acc_{t}', []) for t in ['style', 'artist', 'genre']]
    val_accs = [history.get(f'val_acc_{t}', []) for t in ['style', 'artist', 'genre']]
    
    if all(train_accs) and all(val_accs):
        axes[1].plot(np.mean(train_accs, axis=0), label="Train")
        axes[1].plot(np.mean(val_accs, axis=0), label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Average Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix",
                         output_path: str = None) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
