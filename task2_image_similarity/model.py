"""
Feature extraction model using pretrained ResNet50.

No training required - just feature extraction.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """
    Pretrained ResNet50 feature extractor.
    
    Removes the final classification layer and outputs raw features.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet50
        model = models.resnet50(pretrained=pretrained)
        
        # Remove classification layer, keep everything up to avgpool
        self.features = nn.Sequential(*list(model.children())[:-1])
        
        # Freeze all parameters (no training needed)
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Feature embeddings [B, 2048]
        """
        features = self.features(x)
        features = torch.flatten(features, 1)  # [B, 2048]
        return features


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize embeddings.
    
    Args:
        embeddings: Embedding tensor [B, D] or [N, D]
    
    Returns:
        Normalized embeddings
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)
