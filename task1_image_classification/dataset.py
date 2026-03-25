"""Dataset and label encoding for multi-task image classification."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class LabelEncoder:
    """Bidirectional label encoder."""

    def __init__(self):
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}

    def fit(self, labels: List[str]) -> None:
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def encode(self, label: str) -> int:
        return self.label_to_idx.get(label, -1)

    def decode(self, idx: int) -> str:
        return self.idx_to_label.get(idx, "unknown")

    def get_num_classes(self) -> int:
        return len(self.label_to_idx)

    def __repr__(self) -> str:
        return f"LabelEncoder({len(self.label_to_idx)} classes)"


class WikiArtDataset(Dataset):
    """Multi-task image classification dataset."""

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transforms: Optional[transforms.Compose] = None,
        split: Optional[str] = None,
        return_metadata: bool = False,
        auto_split: bool = True,
        random_seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.return_metadata = return_metadata
        self.random_seed = random_seed

        # Load metadata
        df = pd.read_csv(csv_file, encoding='latin-1')
        
        # Find and validate images
        all_images = self._find_all_images()
        image_path_map = {img.name: img for img in all_images}
        img_col = self._find_image_column(df)
        
        # Build full paths
        if 'relative_path' in df.columns:
            df['_full_path'] = df['relative_path'].apply(lambda x: str(self.root_dir / x))
        elif 'style' in df.columns:
            df['_full_path'] = df.apply(lambda r: str(self.root_dir / r['style'] / r[img_col]), axis=1)
            if not df['_full_path'].apply(lambda x: Path(x).exists()).any():
                df['_full_path'] = df[img_col].apply(lambda x: str(self.root_dir / x))
        else:
            df['_full_path'] = df[img_col].apply(lambda x: str(image_path_map.get(Path(x).name)))
        
        # Filter existing images
        df = df[df['_full_path'].apply(lambda x: x != 'None' and Path(x).exists() if x else False)].copy()
        
        if len(df) == 0:
            raise ValueError("No images found!")

        # Handle split
        if 'split' not in df.columns:
            if auto_split:
                df = self._create_split(df)
            else:
                df['split'] = 'train'
        
        if split is not None:
            df = df[df['split'] == split].copy()

        # Fit encoders
        self.style_encoder = LabelEncoder()
        self.artist_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()

        self.style_encoder.fit(df["style"].unique().tolist())
        self.artist_encoder.fit(df["artist"].unique().tolist())
        self.genre_encoder.fit(df["genre"].unique().tolist())

        self.metadata = df.reset_index(drop=True)
        self.img_col = img_col

    def _find_all_images(self) -> List[Path]:
        ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        images = []
        for e in ext:
            images.extend(self.root_dir.rglob(f'*{e}'))
            images.extend(self.root_dir.rglob(f'*{e.upper()}'))
        return sorted(list(set(images)))

    def _find_image_column(self, df: pd.DataFrame) -> str:
        names = ['image_name', 'image', 'filename', 'file', 'path', 'image_path', 'url', 'img', 'name']
        for col in names:
            if col in df.columns:
                return col
        raise ValueError(f"Image column not found. Options: {', '.join(names)}")

    def _create_split(self, df: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(self.random_seed)
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(df))
        df['split'] = 'train'
        df.loc[df.index.isin(indices[split_idx:]), 'split'] = 'val'
        return df

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        img_path = Path(row['_full_path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))

        if self.transforms:
            image = self.transforms(image)

        output = {
            'image': image,
            'style': torch.tensor(self.style_encoder.encode(row['style']), dtype=torch.long),
            'artist': torch.tensor(self.artist_encoder.encode(row['artist']), dtype=torch.long),
            'genre': torch.tensor(self.genre_encoder.encode(row['genre']), dtype=torch.long),
        }

        if self.return_metadata:
            output['metadata'] = {
                'image_name': row[self.img_col],
                'style_name': row['style'],
                'artist_name': row['artist'],
                'genre_name': row['genre'],
            }

        return output

    def get_label_info(self) -> Dict:
        return {
            'num_styles': self.style_encoder.get_num_classes(),
            'num_artists': self.artist_encoder.get_num_classes(),
            'num_genres': self.genre_encoder.get_num_classes(),
        }


def get_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """Image transformations: augmentation for train, normalization for val."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(
    root_dir: str,
    csv_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    train_split: str = "train",
    val_split: str = "val",
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and validation DataLoaders."""
    train_transforms = get_transforms("train", image_size)
    val_transforms = get_transforms("val", image_size)

    train_dataset = WikiArtDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        transforms=train_transforms,
        split=train_split,
    )

    val_dataset = WikiArtDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        transforms=val_transforms,
        split=val_split,
    )

    label_info = train_dataset.get_label_info()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, label_info
    train_dataset = WikiArtDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        transforms=get_transforms("train", image_size),
        split=train_split,
    )

    val_dataset = WikiArtDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        transforms=get_transforms("val", image_size),
        split=val_split,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    label_info = train_dataset.get_label_info()

    return train_loader, val_loader, label_info



# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Quick test of the dataset."""
    from config import DATASET_CONFIG

    print("\n✓ Testing dataset loader...\n")

    train_loader, val_loader, label_info = create_dataloaders(
        root_dir=DATASET_CONFIG['root_dir'],
        csv_file=DATASET_CONFIG['csv_file'],
        batch_size=DATASET_CONFIG['batch_size'],
        num_workers=0,
    )

    print(f"\n✓ Datasets created successfully!")
    print(f"  - Train: {len(train_loader)} batches")
    print(f"  - Val: {len(val_loader)} batches")
    print(f"  - Styles: {label_info['num_styles']}")
    print(f"  - Artists: {label_info['num_artists']}")
    print(f"  - Genres: {label_info['num_genres']}")

