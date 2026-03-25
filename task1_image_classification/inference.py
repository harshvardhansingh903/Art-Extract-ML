"""Inference module for making predictions on new images."""

from pathlib import Path
from typing import Dict
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import ResNet50MultiTask
from utils import get_device, load_checkpoint


class ImageClassifier:
    """Multi-task image classifier for inference."""
    
    def __init__(self, checkpoint_path: str, label_info: Dict, device: str = "cuda"):
        self.device = get_device(device)
        self.label_info = label_info
        
        self.model = ResNet50MultiTask(
            num_styles=label_info['num_styles'],
            num_artists=label_info['num_artists'],
            num_genres=label_info['num_genres'],
            pretrained=False,
        ).to(self.device)
        
        load_checkpoint(self.model, checkpoint_path, device=str(self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image_path: str) -> Dict:
        """Predict labels for single image."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        predictions = {}
        for task in ['style', 'artist', 'genre']:
            logits = output[task][0]
            probs = torch.softmax(logits, dim=0)
            pred_idx = logits.argmax().item()
            confidence = probs[pred_idx].item()
            
            predictions[task] = {
                'class_idx': pred_idx,
                'confidence': confidence,
            }
        
        return predictions


def main():
    """Example inference usage."""
    from config import PATHS_CONFIG
    
    print("Multi-task Image Classification Inference\n")
    
    # Find checkpoint
    checkpoint_dir = Path(PATHS_CONFIG['checkpoint_dir'])
    checkpoints = list(checkpoint_dir.glob('best_model_*.pth'))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    print(f"Using checkpoint: {checkpoint_path}\n")
    
    # Mock label info (replace with real data)
    label_info = {'num_styles': 25, 'num_artists': 150, 'num_genres': 10}
    
    classifier = ImageClassifier(checkpoint_path, label_info)
    
    print("Classifier ready. Use classifier.predict(image_path) for inference.")


if __name__ == "__main__":
    main()
