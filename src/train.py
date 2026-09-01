import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import create_dataloaders
from model import FoodCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_PATH = OUTPUT_DIR / "history.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    early_stopping_counter = 0
    early_stopping_patience = 5
    min_delta = 1e-4

    num_epochs = 25

    print()
    print(f"Starting training for up to {num_epochs} epochs")
    print(
        f"Early stopping patience: "
        f"{early_stopping_patience} epochs"
    )

    for epoch in range(num_epochs):
        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("=" * 60)

        epoch_lr = optimizer.param_groups[0]["lr"]

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

        # Store metrics for this epoch.
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(epoch_lr)

        # Persist history after every epoch so progress is not lost
        # if training is interrupted.
        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

        # Update learning rate according to validation loss.
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping monitors validation loss.
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0

            print(
                f"Validation loss improved to "
                f"{best_val_loss:.4f}"
            )
        else:
            early_stopping_counter += 1

            print(
                f"No validation loss improvement "
                f"({early_stopping_counter}/"
                f"{early_stopping_patience})"
            )

        # Save the model with the best validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best_model.pth",
            )

            print(
                f"Best model saved "
                f"(val acc: {best_val_accuracy * 100:.2f}%)"
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

        print(
            f"LR    | "
            f"{epoch_lr:.1e} -> {current_lr:.1e}"
        )

        print(f"Train time: {train_time:.1f} seconds")

        if early_stopping_counter >= early_stopping_patience:
            print()
            print(
                f"Early stopping triggered at "
                f"epoch {epoch + 1}."
            )
            break

    print()
    print("=" * 60)
    print("Training completed")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_DIR / 'best_model.pth'}"
    )

    print(
        f"Training history: "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()