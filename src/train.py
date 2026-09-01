import time

import torch
from torch import nn
from tqdm import tqdm

from data import create_dataloaders
from model import FoodCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.perf_counter()

    progress_bar = tqdm(
        loader,
        desc="Training",
        unit="batch",
        dynamic_ncols=True,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()

        total += batch_size

        avg_loss = running_loss / total
        accuracy = correct / total
        learning_rate = optimizer.param_groups[0]["lr"]

        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{accuracy * 100:.2f}%",
            lr=f"{learning_rate:.1e}",
        )

    elapsed_time = time.perf_counter() - start_time

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, elapsed_time


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Validation",
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

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    model = FoodCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    num_epochs = 5

    print()
    print(f"Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("=" * 60)

        train_loss, train_accuracy, train_time = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print()
        print(f"Epoch {epoch + 1}/{num_epochs} completed")
        print(
            f"Train | "
            f"loss: {train_loss:.4f} | "
            f"acc: {train_accuracy * 100:.2f}%"
        )
        print(
            f"Val   | "
            f"loss: {val_loss:.4f} | "
            f"acc: {val_accuracy * 100:.2f}%"
        )
        print(f"Train time: {train_time:.1f} seconds")

    print()
    print("Training completed.")


if __name__ == "__main__":
    main()