"""
dataset.py — LaceDataset for Stage 1 GAN training.

Loads lace fabric images, applies augmentation and normalization
to [-1, 1] as required by GAN discriminators.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class LaceDataset(Dataset):
    """
    PyTorch Dataset for lace fabric images.

    Returns images normalized to [-1, 1] (required by GAN).
    Optionally applies augmentation during training.
    """

    SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        training: bool = True,
    ):
        """
        Args:
            data_dir  : Folder containing lace images (may be nested).
            image_size: Output resolution (square).
            training  : If True, apply random augmentation.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.training = training

        # Collect all image paths
        self.paths = self._collect()
        if len(self.paths) == 0:
            raise ValueError(
                f"No images found in '{data_dir}'. "
                f"Supported: {self.SUPPORTED_EXT}"
            )
        print(f"[Dataset] {len(self.paths)} images found in '{data_dir}'")

        self.transform = self._build_transforms()

    # ------------------------------------------------------------------
    def _collect(self):
        paths = []
        for ext in self.SUPPORTED_EXT:
            paths += list(self.data_dir.rglob(f"*{ext}"))
            paths += list(self.data_dir.rglob(f"*{ext.upper()}"))
        return sorted(set(paths))

    def _build_transforms(self):
        ops = [
            T.Resize((self.image_size, self.image_size),
                     interpolation=T.InterpolationMode.LANCZOS),
        ]

        if self.training:
            ops += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]

        ops += [
            T.ToTensor(),                               # [0,1]
            T.Normalize([0.5, 0.5, 0.5],               # → [-1,1]
                        [0.5, 0.5, 0.5]),
        ]
        return T.Compose(ops)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"[Warning] Cannot load '{self.paths[idx]}': {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))


# ------------------------------------------------------------------
def get_dataloader(
    data_dir: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    training: bool = True,
) -> DataLoader:
    """
    Convenience factory that returns a DataLoader for lace images.

    Args:
        data_dir   : Path to image folder.
        image_size : Square resolution to resize to.
        batch_size : Images per batch.
        num_workers: Parallel loading workers.
        training   : Enable augmentation + shuffle when True.

    Returns:
        torch.utils.data.DataLoader
    """
    ds = LaceDataset(data_dir=data_dir, image_size=image_size,
                     training=training)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,          # Drop incomplete last batch
        persistent_workers=(num_workers > 0),
    )
