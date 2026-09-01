from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import create_dataloaders, create_train_eval_loader
from model import FoodCNNV2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "foodcnn_v2"
    / "best_model.pth"
)


def evaluate(model, loader, criterion, device, description):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            total += batch_size

            avg_loss = running_loss / total
            accuracy = correct / total

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{accuracy * 100:.2f}%",
            )

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_eval_loader = create_train_eval_loader(
        batch_size=32,
        num_workers=0,
    )

    _, val_loader, _ = create_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    model = FoodCNNV2().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    print()

    train_loss, train_accuracy = evaluate(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        device=device,
        description="Train evaluation",
    )

    val_loss, val_accuracy = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        description="Validation",
    )

    print()
    print("=" * 60)
    print("Best model evaluation")
    print("=" * 60)

    print(
        f"Train | loss: {train_loss:.4f} | "
        f"acc: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val   | loss: {val_loss:.4f} | "
        f"acc: {val_accuracy * 100:.2f}%"
    )

    print(
        f"Gap   | "
        f"{(train_accuracy - val_accuracy) * 100:.2f} pp"
    )


if __name__ == "__main__":
    main()