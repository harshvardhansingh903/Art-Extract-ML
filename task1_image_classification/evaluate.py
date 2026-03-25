"""Evaluation of multi-task classification model."""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloaders
from model import ResNet50MultiTask
from config import DATASET_CONFIG, MODEL_CONFIG, DEVICE_CONFIG, PATHS_CONFIG
from utils import get_device, load_checkpoint, plot_confusion_matrix


class Evaluator:
    """Model evaluation wrapper."""
    
    def __init__(self, device: str = "cuda"):
        self.device = get_device(device)
        self.model = None
        self.results = {}
    
    def load_model(self, checkpoint_path: str, label_info: Dict) -> None:
        """Load trained model from checkpoint."""
        self.model = ResNet50MultiTask(
            num_styles=label_info['num_styles'],
            num_artists=label_info['num_artists'],
            num_genres=label_info['num_genres'],
            pretrained=False,
        ).to(self.device)
        load_checkpoint(self.model, checkpoint_path, device=str(self.device))
        self.model.eval()
    
    def evaluate(self, dataloader: DataLoader, task_names: List[str] = None) -> Dict:
        """Evaluate model on dataset."""
        if task_names is None:
            task_names = ["style", "artist", "genre"]
        
        all_preds = {task: [] for task in task_names}
        all_labels = {task: [] for task in task_names}
        
        print("Evaluating...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch["image"].to(self.device)
                output = self.model(images)
                
                for task in task_names:
                    preds = torch.argmax(output[task], dim=1).cpu().numpy()
                    labels = batch[task].cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels)
                
                if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                    print(f"  {(batch_idx + 1) / len(dataloader) * 100:.1f}%")
        
        metrics = {}
        for task in task_names:
            preds = np.array(all_preds[task])
            labels = np.array(all_labels[task])
            metrics[task] = {
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
                "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            }
            print(f"✓ {task.upper()}: acc={metrics[task]['accuracy']:.4f}, f1={metrics[task]['f1']:.4f}")
            all_preds[task] = preds
            all_labels[task] = labels
        
        self.results = {"metrics": metrics, "preds": all_preds, "labels": all_labels}
        return metrics
    
    def plot_confusion_matrices(self, output_dir: str = "./results/plots", 
                               task_names: List[str] = None) -> None:
        """Generate confusion matrices."""
        if task_names is None:
            task_names = list(self.results.get("preds", {}).keys())
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(task_names), figsize=(15, 4))
        if len(task_names) == 1:
            axes = [axes]
        
        for i, task in enumerate(task_names):
            preds = self.results["preds"][task]
            labels = self.results["labels"][task]
            cm = confusion_matrix(labels, preds)
            sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f"{task.upper()}")
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "confusion_matrices.png", dpi=100, bbox_inches="tight")
        plt.close()
    
    def save_results(self, output_path: str = "./results/eval_metrics.json") -> None:
        """Save evaluation results."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "metrics": self.results["metrics"],
            "summary": {
                "mean_accuracy": float(np.mean([m["accuracy"] for m in self.results["metrics"].values()])),
                "mean_f1": float(np.mean([m["f1"] for m in self.results["metrics"].values()])),
            }
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved to {output_path}")


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("TASK 1: EVALUATION")
    print("="*70 + "\n")
    
    device = DEVICE_CONFIG.get("device", "cuda")
    checkpoint_path = Path(PATHS_CONFIG["checkpoint_dir"]) / "best_model_epoch0.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("Loading dataset...")
    _, val_loader, label_info = create_dataloaders(**DATASET_CONFIG)
    print(f"✓ Validation samples: {len(val_loader.dataset)}\n")
    
    evaluator = Evaluator(device=device)
    evaluator.load_model(str(checkpoint_path), label_info)
    
    print("Evaluating...")
    metrics = evaluator.evaluate(val_loader)
    
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrices(output_dir=PATHS_CONFIG["results_dir"] + "/plots")
    
    print("Saving results...")
    evaluator.save_results(PATHS_CONFIG["results_dir"] + "/eval_metrics.json")
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    for task, task_metrics in metrics.items():
        print(f"{task.upper()}: Acc={task_metrics['accuracy']:.4f}, F1={task_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
