from pathlib import Path

import torch
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import v2
from torchvision.datasets import Food101
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_MEAN = [0.5583, 0.4427, 0.3274]
TRAIN_STD = [0.2588, 0.2629, 0.2658]

train_transform = v2.Compose([
    v2.RandomResizedCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
])

eval_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
])

train_base_dataset = Food101(
    root=DATA_DIR,
    split="train",
    download=False,
    transform=train_transform,
)

val_base_dataset = Food101(
    root=DATA_DIR,
    split="train",
    download=False,
    transform=eval_transform,
)

test_base_dataset = Food101(
    root=DATA_DIR,
    split="test",
    download=False,
    transform=eval_transform,
)

indices = list(range(len(train_base_dataset)))
labels = train_base_dataset.labels

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

train_dataset = Subset(train_base_dataset, train_indices)
val_dataset = Subset(val_base_dataset, val_indices)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_base_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)