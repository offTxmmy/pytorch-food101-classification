from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import (
    create_resnet_dataloaders,
    create_resnet_train_eval_loader,
)
from model import (
    create_resnet18,
    enable_partial_finetuning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "resnet18_partial_finetune"
    / "best_model.pth"
)


def evaluate(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
    non_blocking,
    description,
):
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

                loss = criterion(
                    logits,
                    labels,
                )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

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
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"AMP enabled: {amp_enabled}")

    # ========================================================
    # DATA
    # ========================================================

    train_eval_loader = (
        create_resnet_train_eval_loader(
            batch_size=32,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                num_workers > 0
            ),
        )
    )

    _, val_loader, _ = (
        create_resnet_dataloaders(
            batch_size=32,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                num_workers > 0
            ),
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_resnet18(
        num_classes=101,
        freeze_backbone=True,
    )

    model = enable_partial_finetuning(
        model
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    print()
    print("=" * 60)
    print("PARTIAL RESNET18 EVALUATION")
    print("=" * 60)

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    print()

    # ========================================================
    # TRAIN EVALUATION
    # ========================================================

    (
        train_loss,
        train_accuracy,
    ) = evaluate(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        non_blocking=non_blocking,
        description="Train evaluation",
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_loss,
        val_accuracy,
    ) = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        non_blocking=non_blocking,
        description="Validation",
    )

    generalization_gap = (
        train_accuracy
        - val_accuracy
    )

    print()
    print("=" * 60)
    print("BEST MODEL EVALUATION")
    print("=" * 60)

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
        f"Gap   | "
        f"{generalization_gap * 100:.2f} pp"
    )


if __name__ == "__main__":
    main()