"""PyTorch data pipeline for the OCT scans (Step 4).

build_transforms  training augmentation, or resize only for val/test
OCTDataset        one scan -> (image, label, index), rows from the split CSV
make_loaders      train / val / test DataLoaders
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from .data import CLASSES

CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}  # CNV=0, DME=1, DRUSEN=2, NORMAL=3

# The pretrained ResNet expects 3-channel input normalised with ImageNet statistics.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "splits" / "oct_split_fold0.csv"


def build_transforms(train: bool, size: int = 224) -> v2.Compose:
    """Resize every scan to size x size; augment only when training.

    Plain resize (no padding): padding would leave black bars on every scan
    except the 512x512 ones, and those are all DME, a giveaway the model
    could learn instead of the disease.
    """
    steps = [
        v2.Grayscale(num_output_channels=3),  # copy the grey channel 3 times
        v2.Resize((size, size), antialias=True),
    ]
    if train:
        steps += [
            # Crop after resizing, so the crop's aspect ratio applies to the square image
            v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(),  # never vertical: a retina is never upside down
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),  # uint8 0-255 -> float 0-1
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(steps)


class OCTDataset(Dataset):
    """Scans of one split. Returns (image, label, index).

    `index` is the row in self.table, so later steps can find which scan
    (path, patient) a prediction belongs to.
    """

    def __init__(self, image_root, split: str, split_csv=DEFAULT_SPLIT, transform=None):
        table = pd.read_csv(split_csv, dtype={"patient": str})
        assert split in set(table["split"]), f"no rows with split={split!r} in {split_csv}"
        self.table = table[table["split"] == split].reset_index(drop=True)
        self.labels = torch.tensor(self.table["label"].map(CLASS_TO_INDEX).to_numpy())
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        path = self.image_root / self.table.at[index, "path"]
        image = decode_image(str(path), mode=ImageReadMode.GRAY)  # (1, H, W) uint8
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], index


def make_loaders(image_root, split_csv=DEFAULT_SPLIT, batch_size: int = 64,
                 num_workers: int = 4, size: int = 224) -> dict[str, DataLoader]:
    """Train is shuffled and augmented; val and test keep their order and are only resized."""
    loaders = {}
    for split in ("train", "val", "test"):
        train = split == "train"
        dataset = OCTDataset(image_root, split, split_csv, build_transforms(train, size))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders