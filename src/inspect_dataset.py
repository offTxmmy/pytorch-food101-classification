from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = Food101(
        root=DATA_DIR,
        split="train",
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    print("Number of training samples:", len(train_dataset))

    image, label = train_dataset[0]

    print("\nSingle sample")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Label:", label)
    print("Class name:", train_dataset.classes[label])

    images, labels = next(iter(train_loader))

    print("\nBatch")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Images dtype:", images.dtype)
    print("Labels dtype:", labels.dtype)


if __name__ == "__main__":
    main()