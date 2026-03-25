"""Outlier detection for predictions with low confidence."""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from model import ResNet50MultiTask
from config import DATASET_CONFIG, MODEL_CONFIG, DEVICE_CONFIG, PATHS_CONFIG
from utils import get_device, load_checkpoint


class OutlierDetector:
    """Detects outliers by confidence threshold."""
    
    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        self.device = get_device(device)
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.outliers = {}
        self.statistics = {}
    
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
    
    def detect_outliers(self, dataloader: DataLoader, task_names: List[str] = None) -> Dict:
        """Detect outliers by confidence threshold."""
        if task_names is None:
            task_names = ["style", "artist", "genre"]
        
        outliers = {task: [] for task in task_names}
        confidence_scores = {task: [] for task in task_names}
        
        print(f"Detecting outliers (threshold: {self.confidence_threshold})...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch["image"].to(self.device)
                output = self.model(images)
                
                for task in task_names:
                    logits = output[task]
                    probs = F.softmax(logits, dim=1)
                    confidences, preds = torch.max(probs, dim=1)
                    true_labels = batch[task]
                    
                    confidence_scores[task].extend(confidences.cpu().numpy())
                    
                    for i in range(len(images)):
                        if confidences[i].item() < self.confidence_threshold:
                            outliers[task].append({
                                "confidence": float(confidences[i].item()),
                                "pred_idx": int(preds[i].item()),
                                "true_idx": int(true_labels[i].item()),
                                "is_correct": bool(preds[i] == true_labels[i]),
                            })
                
                if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                    print(f"  {(batch_idx + 1) / len(dataloader) * 100:.1f}%")
        
        # Compute statistics
        for task in task_names:
            scores = np.array(confidence_scores[task])
            self.statistics[task] = {
                "total": len(scores),
                "outliers": len(outliers[task]),
                "outlier_pct": float(len(outliers[task]) / len(scores) * 100) if len(scores) > 0 else 0,
                "mean_conf": float(np.mean(scores)),
                "std_conf": float(np.std(scores)),
                "min_conf": float(np.min(scores)),
                "max_conf": float(np.max(scores)),
                "median_conf": float(np.median(scores)),
            }
        
        self.outliers = outliers
        return outliers
    
    def save_results(self, output_dir: str = "./results") -> None:
        """Save outlier detection results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / "outliers.json"
        with open(output_path, "w") as f:
            json.dump({"outliers": self.outliers, "statistics": self.statistics}, f, indent=2)
        print(f"✓ Results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "="*70)
        print("OUTLIER DETECTION SUMMARY")
        print("="*70)
        for task, stats in self.statistics.items():
            print(f"\n{task.upper()}:")
            print(f"  Total samples: {stats['total']}")
            print(f"  Outliers: {stats['outliers']} ({stats['outlier_pct']:.2f}%)")
            print(f"  Mean confidence: {stats['mean_conf']:.4f}")
            print(f"  Median confidence: {stats['median_conf']:.4f}")
    
    def get_low_confidence_samples(self, task: str, top_n: int = 10) -> List[Dict]:
        """Get lowest confidence samples."""
        return sorted(self.outliers.get(task, []), key=lambda x: x["confidence"])[:top_n]


def main():
    """Main outlier detection pipeline."""
    print("="*70)
    print("TASK 1: OUTLIER DETECTION")
    print("="*70 + "\n")
    
    device = DEVICE_CONFIG.get("device", "cuda")
    checkpoint_path = Path(PATHS_CONFIG["checkpoint_dir"]) / "best_model_epoch0.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("Loading dataset...")
    _, val_loader, label_info = create_dataloaders(**DATASET_CONFIG)
    print(f"✓ Validation samples: {len(val_loader.dataset)}\n")
    
    detector = OutlierDetector(device=device, confidence_threshold=0.5)
    detector.load_model(str(checkpoint_path), label_info)
    
    print("Detecting outliers...")
    detector.detect_outliers(val_loader)
    detector.print_summary()
    
    print("\nLowest confidence samples per task:")
    for task in ["style", "artist", "genre"]:
        low_conf = detector.get_low_confidence_samples(task, top_n=3)
        print(f"\n  {task.upper()}:")
        for i, sample in enumerate(low_conf, 1):
            print(f"    {i}. Confidence: {sample['confidence']:.4f}, "
                  f"Pred: {sample['pred_idx']}, Correct: {sample['is_correct']}")
    
    print("\nSaving results...")
    detector.save_results(PATHS_CONFIG["results_dir"])
    print()


if __name__ == "__main__":
    main()
