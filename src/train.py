import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import create_resnet_dataloaders
from model import create_resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "resnet18_frozen"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / EXPERIMENT_NAME

CHECKPOINT_PATH = EXPERIMENT_DIR / "best_model.pth"
HISTORY_PATH = EXPERIMENT_DIR / "history.json"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled,
    non_blocking,
):
    model.train()

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

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
        images = images.to(
            device,
            non_blocking=non_blocking,
        )
        labels = labels.to(
            device,
            non_blocking=non_blocking,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
    non_blocking,
):
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

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=non_blocking,
            )
            labels = labels.to(
                device,
                non_blocking=non_blocking,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
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

    amp_enabled = device.type == "cuda"
    pin_memory = device.type == "cuda"
    non_blocking = device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        num_workers = 4
    else:
        num_workers = 0

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"AMP enabled: {amp_enabled}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")

    train_loader, val_loader, _ = create_resnet_dataloaders(
        batch_size=32,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    model = create_resnet18(
        num_classes=101,
        freeze_backbone=True,
    ).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=1e-3,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
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
    early_stopping_patience = 3
    min_delta = 1e-4

    num_epochs = 10

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
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            non_blocking=non_blocking,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            non_blocking=non_blocking,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(epoch_lr)

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=4,
            )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint = {
                "experiment": EXPERIMENT_NAME,
                "freeze_backbone": True,
                "num_classes": 101,
                "amp_enabled": amp_enabled,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"Best model saved "
                f"(val acc: "
                f"{best_val_accuracy * 100:.2f}%)"
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

        print(
            f"Train time: "
            f"{train_time:.1f} seconds"
        )

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
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Training history: "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()