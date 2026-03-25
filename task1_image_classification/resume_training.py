"""Resume training from a specific epoch checkpoint."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional

from dataset import create_dataloaders
from model import ResNet50MultiTask, count_parameters
from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LOSS_CONFIG, DEVICE_CONFIG, PATHS_CONFIG
from utils import setup_directories, get_device, save_checkpoint, save_metrics_json, format_metrics_for_logging
from train import Trainer


def load_checkpoint_for_resume(checkpoint_path: str, device: torch.device):
    """
    Load checkpoint and extract metadata for resuming training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device for loading
    
    Returns:
        Tuple of (model_state, optimizer_state, start_epoch, best_val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✓ Checkpoint loaded: Epoch {checkpoint['epoch']}")
    print(f"  - Best val loss so far: {best_val_loss:.4f}")
    print(f"  - Resuming from epoch: {start_epoch}\n")
    
    return checkpoint['model_state_dict'], checkpoint.get('optimizer_state_dict'), start_epoch, best_val_loss


def resume_training_from_epoch(
    resume_from_epoch: int,
    train_until_epoch: int,
    device: Optional[torch.device] = None
):
    """
    Resume training from a specific epoch.
    
    Args:
        resume_from_epoch: Epoch to load checkpoint from (e.g., 2)
        train_until_epoch: Final epoch to train until (e.g., 3)
        device: Device to use ('cuda' or 'cpu')
    """
    device = device or get_device(DEVICE_CONFIG['device'])
    
    print("\n" + "="*70)
    print("TASK 1: RESUME TRAINING")
    print("="*70 + "\n")
    
    # Get absolute path for checkpoint directory
    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = script_dir / PATHS_CONFIG['checkpoint_dir']
    
    # Check if checkpoint exists
    checkpoint_candidates = [
        f"best_model_epoch{resume_from_epoch}.pth",
        f"model_epoch{resume_from_epoch}.pth",
    ]
    
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        full_path = checkpoint_dir / candidate
        if full_path.exists():
            checkpoint_path = str(full_path)
            break
    
    if not checkpoint_path:
        # If exact epoch not found, suggest alternatives
        available_checkpoints = [f.name for f in checkpoint_dir.glob("*.pth")]
        print(f"⚠ Checkpoint for Epoch {resume_from_epoch} not found!")
        print(f"  Available checkpoints: {available_checkpoints}\n")
        
        # Use first available checkpoint if none found
        if available_checkpoints:
            checkpoint_path = str(checkpoint_dir / available_checkpoints[0])
            print(f"→ Using: {available_checkpoints[0]}\n")
        else:
            print("✗ No checkpoints found! Train from scratch instead.\n")
            return
    
    print("Loading dataset...")
    # Filter DATASET_CONFIG to only include valid arguments for create_dataloaders
    valid_keys = {'root_dir', 'csv_file', 'batch_size', 'num_workers', 'image_size', 'train_split', 'val_split'}
    dataset_args = {k: v for k, v in DATASET_CONFIG.items() if k in valid_keys}
    
    train_loader, val_loader, label_info = create_dataloaders(**dataset_args)
    print(f"✓ Dataset loaded: {len(train_loader)} train batches, {len(val_loader)} val batches\n")
    
    # Create model
    print("Initializing model...")
    model = ResNet50MultiTask(
        num_styles=label_info['num_styles'],
        num_artists=label_info['num_artists'],
        num_genres=label_info['num_genres']
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"✓ Model ready: {total_params:,} total parameters ({trainable_params:,} trainable)\n")
    
    # Load checkpoint
    model_state, optimizer_state, start_epoch, best_val_loss = load_checkpoint_for_resume(
        checkpoint_path, device
    )
    model.load_state_dict(model_state)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, TRAINING_CONFIG)
    trainer.best_val_loss = best_val_loss
    
    # Fix loss weights format (convert from config format)
    loss_weights = {
        'style': LOSS_CONFIG.get('style_weight', 1.0),
        'artist': LOSS_CONFIG.get('artist_weight', 1.0),
        'genre': LOSS_CONFIG.get('genre_weight', 1.0),
    }
    trainer.loss_weights = loss_weights
    
    # Load optimizer state if available
    if optimizer_state:
        trainer.optimizer.load_state_dict(optimizer_state)
        print("✓ Optimizer state restored\n")
    
    # Train from start_epoch until train_until_epoch
    num_epochs_to_train = train_until_epoch - start_epoch + 1
    
    print(f"Training plan: Epoch {start_epoch} → Epoch {train_until_epoch} ({num_epochs_to_train} epochs)\n")
    
    print("="*70)
    print("TRAINING RESUMED")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, train_until_epoch + 1):
        print(f"Epoch {epoch}/{train_until_epoch}")
        
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.validate()
        
        print(f"{format_metrics_for_logging(train_metrics, 'TRAIN')}")
        print(f"{format_metrics_for_logging(val_metrics, 'VAL')}\n")
        
        # Update history
        for task in ['style', 'artist', 'genre']:
            trainer.history[f'train_acc_{task}'].append(train_metrics[f'acc_{task}'])
            trainer.history[f'val_acc_{task}'].append(val_metrics[f'acc_{task}'])
        trainer.history['train_loss'].append(train_metrics['loss'])
        trainer.history['val_loss'].append(val_metrics['loss'])
        
        # Save checkpoint
        if val_metrics['loss'] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics['loss']
            trainer.patience_counter = 0
            save_checkpoint(model, trainer.optimizer, epoch, val_metrics, 
                          str(checkpoint_dir), f'best_model_epoch{epoch}')
            print(f"✓ Best model saved: best_model_epoch{epoch}.pth (val_loss: {trainer.best_val_loss:.4f})\n")
        else:
            trainer.patience_counter += 1
    
    print("="*70)
    print("TRAINING COMPLETED")
    print("="*70 + "\n")
    
    # Save final metrics
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / PATHS_CONFIG['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    save_metrics_json(trainer.history, str(results_dir / 'metrics.json'))
    print(f"✓ Metrics saved to {results_dir / 'metrics.json'}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume training from specific epoch")
    parser.add_argument("--resume-from", type=int, default=2, help="Epoch to resume from (default: 2)")
    parser.add_argument("--train-until", type=int, default=3, help="Final epoch to train until (default: 3)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    
    args = parser.parse_args()
    
    resume_training_from_epoch(
        resume_from_epoch=args.resume_from,
        train_until_epoch=args.train_until,
        device=args.device
    )
