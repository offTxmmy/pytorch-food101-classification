from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

stats_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

base_dataset = Food101(
    root=DATA_DIR,
    split="train",
    download=False,
    transform=stats_transform,
)

indices = list(range(len(base_dataset)))
labels = base_dataset.labels

train_indices, _ = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

stats_dataset = Subset(base_dataset, train_indices)

stats_loader = DataLoader(
    stats_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

channel_sum = torch.zeros(3)
channel_squared_sum = torch.zeros(3)
num_pixels = 0

for images, _ in stats_loader:
    channel_sum += images.sum(dim=(0, 2, 3))
    channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

    num_pixels += (
        images.shape[0]
        * images.shape[2]
        * images.shape[3]
    )

mean = channel_sum / num_pixels

std = torch.sqrt(
    channel_squared_sum / num_pixels - mean ** 2
)

print("Mean:", mean)
print("Std:", std)