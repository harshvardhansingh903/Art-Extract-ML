"""Multi-task image classification trainer."""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from model import ResNet50MultiTask, count_parameters
from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LOSS_CONFIG, DEVICE_CONFIG, PATHS_CONFIG
from utils import setup_directories, get_device, save_checkpoint, save_metrics_json, format_metrics_for_logging


class Trainer:
    """Multi-task training wrapper."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 device: torch.device, config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                     weight_decay=config['weight_decay'])
        self.criterion = nn.CrossEntropyLoss()
        self.loss_weights = LOSS_CONFIG
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        setup_directories(PATHS_CONFIG)
        
        self.history = {
            'train_loss': [], 'train_acc_style': [], 'train_acc_artist': [], 'train_acc_genre': [],
            'val_loss': [], 'val_acc_style': [], 'val_acc_artist': [], 'val_acc_genre': [],
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_counts = {'style': 0, 'artist': 0, 'genre': 0}
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = {task: batch[task].to(self.device) for task in ['style', 'artist', 'genre']}
            
            self.optimizer.zero_grad()
            output = self.model(images)
            
            loss = sum(self.loss_weights[task] * self.criterion(output[task], labels[task]) 
                      for task in ['style', 'artist', 'genre'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            for task in ['style', 'artist', 'genre']:
                correct_counts[task] += (output[task].argmax(dim=1) == labels[task]).sum().item()
            total_samples += images.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  [{batch_idx + 1}/{len(self.train_loader)}] Loss: {total_loss / (batch_idx + 1):.4f}")
        
        return {
            'loss': total_loss / len(self.train_loader),
            **{f'acc_{task}': correct_counts[task] / total_samples for task in ['style', 'artist', 'genre']}
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct_counts = {'style': 0, 'artist': 0, 'genre': 0}
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = {task: batch[task].to(self.device) for task in ['style', 'artist', 'genre']}
                
                output = self.model(images)
                loss = sum(self.loss_weights[task] * self.criterion(output[task], labels[task]) 
                          for task in ['style', 'artist', 'genre'])
                
                total_loss += loss.item()
                for task in ['style', 'artist', 'genre']:
                    correct_counts[task] += (output[task].argmax(dim=1) == labels[task]).sum().item()
                total_samples += images.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            **{f'acc_{task}': correct_counts[task] / total_samples for task in ['style', 'artist', 'genre']}
        }
    
    def fit(self, num_epochs: int) -> Dict:
        """Train for multiple epochs."""
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"{format_metrics_for_logging(train_metrics, 'TRAIN')}")
            print(f"{format_metrics_for_logging(val_metrics, 'VAL')}\n")
            
            for task in ['style', 'artist', 'genre']:
                self.history[f'train_acc_{task}'].append(train_metrics[f'acc_{task}'])
                self.history[f'val_acc_{task}'].append(val_metrics[f'acc_{task}'])
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, 
                              PATHS_CONFIG['checkpoint_dir'], 'best_model')
                print(f"✓ Best model saved (val_loss: {self.best_val_loss:.4f})\n")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        print("="*70)
        print("TRAINING COMPLETED")
        print("="*70 + "\n")
        return self.history


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("TASK 1: MULTI-TASK IMAGE CLASSIFICATION")
    print("="*70 + "\n")
    
    device = get_device(DEVICE_CONFIG['device'])
    print(f"✓ Device ready\n")
    
    print("Loading dataset...")
    train_loader, val_loader, label_info = create_dataloaders(
        root_dir=DATASET_CONFIG['root_dir'],
        csv_file=DATASET_CONFIG['csv_file'],
        batch_size=DATASET_CONFIG['batch_size'],
        num_workers=DATASET_CONFIG['num_workers'],
        image_size=DATASET_CONFIG['image_size'],
    )
    print(f"✓ Dataset loaded\n")
    
    print("Initializing model...")
    model = ResNet50MultiTask(
        num_styles=label_info['num_styles'],
        num_artists=label_info['num_artists'],
        num_genres=label_info['num_genres'],
        pretrained=MODEL_CONFIG['pretrained'],
        freeze_backbone=MODEL_CONFIG['freeze_backbone'],
        dropout_rate=MODEL_CONFIG['dropout_rate'],
    ).to(device)
    
    total, trainable = count_parameters(model)
    print(f"✓ Model initialized ({trainable:,} trainable parameters)\n")
    
    trainer = Trainer(model, train_loader, val_loader, device, TRAINING_CONFIG)
    history = trainer.fit(num_epochs=TRAINING_CONFIG['num_epochs'])
    
    print("Saving results...")
    save_metrics_json(history, str(Path(PATHS_CONFIG['results_dir']) / 'metrics.json'))
    print(f"✓ Results saved\n")
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    for task in ['style', 'artist', 'genre']:
        print(f"Final {task} accuracy: {history[f'train_acc_{task}'][-1]:.4f}")


if __name__ == "__main__":
    main()
