"""
Generate comprehensive results for Task 1: Multi-task Image Classification.

Loads trained model, evaluates on validation set, and creates
visualizations and analysis ready for GitHub submission.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from model import ResNet50MultiTask
from dataset import create_dataloaders
from config import DATASET_CONFIG, MODEL_CONFIG, DEVICE_CONFIG, PATHS_CONFIG
from utils import get_device, load_checkpoint


class ResultsGenerator:
    """Generate comprehensive evaluation results."""
    
    def __init__(self, device: str = "cpu"):
        self.device = get_device(device)
        self.model = None
        self.label_encoders = {}
        self.results = {}
        self.predictions_data = []
        self.outliers_data = []
    
    def setup_output_dirs(self):
        """Create output directories."""
        Path(PATHS_CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)
        Path(PATHS_CONFIG["results_dir"]) / "plots" 
        Path(PATHS_CONFIG["results_dir"]) / "predictions"
        Path(PATHS_CONFIG["results_dir"]) / "outliers"
        
        for subdir in ["plots", "predictions", "outliers"]:
            (Path(PATHS_CONFIG["results_dir"]) / subdir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_data(self, checkpoint_path: str):
        """Load model and create dataloaders."""
        print("\n" + "="*70)
        print("LOADING MODEL AND DATA")
        print("="*70)
        
        # Create dataloaders to get label info
        train_loader, val_loader, label_info = create_dataloaders(
            root_dir=DATASET_CONFIG["root_dir"],
            csv_file=DATASET_CONFIG["csv_file"],
            batch_size=DATASET_CONFIG["batch_size"],
            num_workers=0,
        )
        
        # Get encoders from validation dataset and enable metadata
        val_dataset = val_loader.dataset
        val_dataset.return_metadata = True
        
        # Recreate loader with metadata
        from torch.utils.data import DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=DATASET_CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        
        # Extract encoders
        self.label_encoders = {
            "style": val_dataset.style_encoder,
            "artist": val_dataset.artist_encoder,
            "genre": val_dataset.genre_encoder,
        }
        
        # Load model
        # Filter MODEL_CONFIG for valid ResNet50MultiTask parameters
        valid_keys = {"pretrained", "freeze_backbone", "dropout_rate"}
        model_params = {k: v for k, v in MODEL_CONFIG.items() if k in valid_keys}
        
        self.model = ResNet50MultiTask(
            num_styles=label_info["num_styles"],
            num_artists=label_info["num_artists"],
            num_genres=label_info["num_genres"],
            **model_params
        ).to(self.device)
        
        load_checkpoint(self.model, checkpoint_path, device=str(self.device))
        self.model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Classes - Style: {label_info['num_styles']}, "
              f"Artist: {label_info['num_artists']}, Genre: {label_info['num_genres']}")
        
        return val_loader, label_info
    
    def evaluate_on_validation(self, val_loader):
        """Evaluate model on validation set."""
        print("\n" + "="*70)
        print("EVALUATING ON VALIDATION SET")
        print("="*70)
        
        all_preds = {"style": [], "artist": [], "genre": []}
        all_labels = {"style": [], "artist": [], "genre": []}
        all_confidence = {"style": [], "artist": [], "genre": []}
        all_images = []
        
        print("Processing validation images...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(self.device)
                outputs = self.model(images)
                
                # Get image names from metadata
                metadata = batch.get("metadata", {})
                image_names = metadata.get("image_name", ["unknown"] * images.shape[0])
                if not isinstance(image_names, list):
                    image_names = image_names.tolist() if hasattr(image_names, "tolist") else [str(i) for i in range(images.shape[0])]
                
                for task in ["style", "artist", "genre"]:
                    probs = torch.softmax(outputs[task], dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    confidence = torch.max(probs, dim=1)[0].cpu().numpy()
                    labels = batch[task].cpu().numpy()
                    
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels)
                    all_confidence[task].extend(confidence)
                
                all_images.extend(image_names)
                
                if (batch_idx + 1) % max(1, len(val_loader) // 5) == 0:
                    print(f"  {(batch_idx + 1) / len(val_loader) * 100:.1f}%")
        
        # Compute metrics
        metrics = {}
        for task in ["style", "artist", "genre"]:
            preds = np.array(all_preds[task])
            labels = np.array(all_labels[task])
            
            metrics[task] = {
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
                "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            }
            
            print(f"✓ {task.upper():8} | Accuracy: {metrics[task]['accuracy']:.4f} | "
                  f"F1: {metrics[task]['f1']:.4f} | Precision: {metrics[task]['precision']:.4f}")
        
        self.results = {
            "metrics": metrics,
            "preds": all_preds,
            "labels": all_labels,
            "confidence": all_confidence,
            "images": all_images,
        }
        
        return metrics
    
    def identify_outliers(self, threshold: float = 0.5):
        """Identify low-confidence predictions as outliers."""
        print("\n" + "="*70)
        print("IDENTIFYING OUTLIERS")
        print("="*70)
        
        outliers = []
        
        for idx in range(len(self.results["images"])):
            for task in ["style", "artist", "genre"]:
                if self.results["confidence"][task][idx] < threshold:
                    pred_idx = self.results["preds"][task][idx]
                    label_idx = self.results["labels"][task][idx]
                    
                    outlier = {
                        "index": idx,
                        "image": self.results["images"][idx],
                        "task": task,
                        "predicted": self.label_encoders[task].decode(int(pred_idx)),
                        "actual": self.label_encoders[task].decode(int(label_idx)),
                        "confidence": float(self.results["confidence"][task][idx]),
                    }
                    outliers.append(outlier)
        
        # Sort by confidence
        outliers = sorted(outliers, key=lambda x: x["confidence"])
        
        self.outliers_data = outliers
        print(f"✓ Found {len(outliers)} outlier predictions (confidence < {threshold})")
        
        if outliers:
            print(f"  Most uncertain prediction confidence: {outliers[0]['confidence']:.4f}")
        
        return outliers
    
    def generate_sample_predictions(self, num_samples: int = 10):
        """Generate sample predictions for visualization."""
        print("\n" + "="*70)
        print("GENERATING SAMPLE PREDICTIONS")
        print("="*70)
        
        samples = []
        indices = np.random.choice(len(self.results["images"]), 
                                  min(num_samples, len(self.results["images"])), 
                                  replace=False)
        
        for idx in indices:
            sample = {
                "image": self.results["images"][idx],
                "predictions": {},
            }
            
            for task in ["style", "artist", "genre"]:
                pred_idx = self.results["preds"][task][idx]
                label_idx = self.results["labels"][task][idx]
                confidence = self.results["confidence"][task][idx]
                
                sample["predictions"][task] = {
                    "predicted": self.label_encoders[task].decode(pred_idx),
                    "actual": self.label_encoders[task].decode(label_idx),
                    "confidence": float(confidence),
                    "correct": pred_idx == label_idx,
                }
            
            samples.append(sample)
        
        self.predictions_data = samples
        print(f"✓ Generated {len(samples)} sample predictions")
        
        return samples
    
    def plot_confusion_matrices(self):
        """Generate and save confusion matrix plots."""
        print("\n" + "="*70)
        print("GENERATING CONFUSION MATRICES")
        print("="*70)
        
        output_dir = Path(PATHS_CONFIG["results_dir"]) / "plots"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
        
        for i, task in enumerate(["style", "artist", "genre"]):
            cm = np.array(self.results["metrics"][task]["confusion_matrix"])
            
            # Plot
            sns.heatmap(cm, ax=axes[i], cmap="Blues", annot=False, fmt="d", cbar=True)
            axes[i].set_title(f"{task.upper()} Confusion Matrix", fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Saved confusion matrices to: {output_dir / 'confusion_matrices.png'}")
    
    def plot_metrics_comparison(self):
        """Generate metrics comparison bar chart."""
        print("Generating metrics comparison chart...")
        
        output_dir = Path(PATHS_CONFIG["results_dir"]) / "plots"
        
        tasks = ["style", "artist", "genre"]
        metrics_names = ["accuracy", "precision", "recall", "f1"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            values = [self.results["metrics"][task][metric] for task in tasks]
            
            bars = axes[i].bar(tasks, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            axes[i].set_ylabel(metric.capitalize(), fontsize=11)
            axes[i].set_title(f"{metric.capitalize()} Comparison", fontsize=12, fontweight="bold")
            axes[i].set_ylim([0, 1.0])
            axes[i].grid(axis="y", alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Saved metrics chart to: {output_dir / 'metrics_comparison.png'}")
    
    def save_metrics_json(self):
        """Save metrics as JSON."""
        output_dir = Path(PATHS_CONFIG["results_dir"])
        
        metrics_output = {
            "metadata": {
                "model": "ResNet50MultiTask",
                "dataset": "WikiArt",
                "device": str(self.device),
            },
            "metrics": {}
        }
        
        for task in ["style", "artist", "genre"]:
            metrics = self.results["metrics"][task].copy()
            # Remove confusion matrix for JSON (convert to file)
            cm = metrics.pop("confusion_matrix")
            metrics_output["metrics"][task] = metrics
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_output, f, indent=2)
        
        print(f"✓ Saved metrics to: {output_dir / 'metrics.json'}")
    
    def save_outliers_visualization(self, num_show: int = 10):
        """Create visualization of outlier predictions (text-based)."""
        print(f"Creating outlier report ({num_show} items)...")
        
        output_dir = Path(PATHS_CONFIG["results_dir"]) / "outliers"
        
        # Show top outliers
        outliers_to_show = self.outliers_data[:num_show]
        
        # Create text summary
        summary_text = "LOW CONFIDENCE PREDICTIONS (OUTLIERS)\n"
        summary_text += "=" * 80 + "\n\n"
        
        for i, outlier in enumerate(outliers_to_show, 1):
            summary_text += f"{i}. Image: {outlier['image']}\n"
            summary_text += f"   Task: {outlier['task'].upper()}\n"
            summary_text += f"   Predicted: {outlier['predicted']}\n"
            summary_text += f"   Actual: {outlier['actual']}\n"
            summary_text += f"   Confidence: {outlier['confidence']:.4f}\n"
            summary_text += "-" * 80 + "\n"
        
        # Save to file
        with open(output_dir / "outliers.txt", "w") as f:
            f.write(summary_text)
        
        print(f"✓ Saved outlier report to: {output_dir / 'outliers.txt'}")
    
    def save_predictions_visualization(self):
        """Create visualization of sample predictions (text-based)."""
        print("Creating predictions report...")
        
        output_dir = Path(PATHS_CONFIG["results_dir"]) / "predictions"
        
        # Create text summary
        summary_text = "SAMPLE PREDICTIONS\n"
        summary_text += "=" * 100 + "\n\n"
        
        for i, sample in enumerate(self.predictions_data, 1):
            summary_text += f"{i}. Image: {sample['image']}\n"
            
            for task in ["style", "artist", "genre"]:
                pred = sample["predictions"][task]
                status = "✓" if pred["correct"] else "✗"
                summary_text += (f"   {task.upper()}: {status}\n"
                               f"     Predicted: {pred['predicted']}\n"
                               f"     Actual: {pred['actual']}\n"
                               f"     Confidence: {pred['confidence']:.4f}\n")
            
            summary_text += "-" * 100 + "\n"
        
        # Save to file
        with open(output_dir / "predictions.txt", "w") as f:
            f.write(summary_text)
        
        print(f"✓ Saved predictions report to: {output_dir / 'predictions.txt'}")
    
    def print_summary(self):
        """Print summary report."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print("\n📊 METRICS:")
        for task in ["style", "artist", "genre"]:
            m = self.results["metrics"][task]
            print(f"\n{task.upper()}:")
            print(f"  Accuracy:  {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1-Score:  {m['f1']:.4f}")
        
        print(f"\n📍 OUTLIERS: {len(self.outliers_data)} predictions with confidence < 0.5")
        
        print(f"\n📈 PREDICTIONS: {len(self.predictions_data)} samples shown")
        
        print("\n✓ All results saved to: " + str(Path(PATHS_CONFIG["results_dir"])))
        print("="*70 + "\n")


def main():
    """Main execution."""
    # Setup
    generator = ResultsGenerator(device=DEVICE_CONFIG.get("device", "cpu"))
    generator.setup_output_dirs()
    
    # Find best checkpoint
    checkpoint_dir = Path(PATHS_CONFIG["checkpoint_dir"])
    checkpoints = sorted(checkpoint_dir.glob("best_model_*.pth"))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    best_checkpoint = str(checkpoints[-1])
    print(f"\n✓ Using checkpoint: {best_checkpoint}")
    
    # Load and evaluate
    val_loader, label_info = generator.load_model_and_data(best_checkpoint)
    generator.evaluate_on_validation(val_loader)
    
    # Generate outputs
    generator.identify_outliers(threshold=0.5)
    generator.generate_sample_predictions(num_samples=10)
    
    # Save visualizations
    generator.plot_confusion_matrices()
    generator.plot_metrics_comparison()
    generator.save_metrics_json()
    generator.save_outliers_visualization(num_show=10)
    generator.save_predictions_visualization()
    
    # Print summary
    generator.print_summary()


if __name__ == "__main__":
    main()
