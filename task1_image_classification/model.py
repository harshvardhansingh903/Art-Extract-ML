"""ResNet50 multi-task image classification model."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50MultiTask(nn.Module):
    """
    ResNet50 multi-task classifier for style, artist, and genre.
    
    Shares backbone features across three classification heads for independent task predictions.
    """
    
    def __init__(
        self,
        num_styles: int,
        num_artists: int,
        num_genres: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.num_styles = num_styles
        self.num_artists = num_artists
        self.num_genres = num_genres
        
        # Backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ============================================================
        # Classification heads (shared architecture)
        self.style_head = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_styles),
        )
        
        self.artist_head = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_artists),
        )
        
        self.genre_head = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_genres),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-task outputs."""
        features = self.backbone(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        return {
            'style': self.style_head(features),
            'artist': self.artist_head(features),
            'genre': self.genre_head(features),
            'features': features,
        }
    
    def get_feature_dim(self) -> int:
        """Get feature vector dimension."""
        return 2048


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

