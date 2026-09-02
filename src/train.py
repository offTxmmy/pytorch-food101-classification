import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import create_resnet_dataloaders
from model import create_resnet18, enable_partial_finetuning


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "resnet18_partial_finetune"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / EXPERIMENT_NAME

CHECKPOINT_PATH = EXPERIMENT_DIR / "best_model.pth"
HISTORY_PATH = EXPERIMENT_DIR / "history.json"

SOURCE_CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "experiments"
    / "resnet18_frozen"
    / "best_model.pth"
)

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BATCHNORM POLICY
# ============================================================

def set_frozen_batchnorm_eval(model):
    """
    Keep BatchNorm layers belonging to the frozen backbone
    in eval mode.

    layer4 BatchNorm layers are intentionally left untouched
    so that they remain in train mode during partial fine-tuning.
    """

    model.bn1.eval()

    for layer in (
        model.layer1,
        model.layer2,
        model.layer3,
    ):
        for module in layer.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


# ============================================================
# TRAINING
# ============================================================

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

    # model.train() sets every BatchNorm to training mode.
    # Restore eval mode only for the frozen part.
    set_frozen_batchnorm_eval(model)

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

        optimizer.zero_grad(
            set_to_none=True
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

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

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

        layer4_lr = (
            optimizer.param_groups[0]["lr"]
        )

        fc_lr = (
            optimizer.param_groups[1]["lr"]
        )

        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{accuracy * 100:.2f}%",
            lr4=f"{layer4_lr:.1e}",
            lrfc=f"{fc_lr:.1e}",
        )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    avg_loss = running_loss / total
    accuracy = correct / total

    return (
        avg_loss,
        accuracy,
        elapsed_time,
    )


# ============================================================
# VALIDATION
# ============================================================

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

            avg_loss = (
                running_loss / total
            )

            accuracy = (
                correct / total
            )

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=(
                    f"{accuracy * 100:.2f}%"
                ),
            )

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
    )

    pin_memory = (
        device.type == "cuda"
    )

    non_blocking = (
        device.type == "cuda"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        num_workers = 4
    else:
        num_workers = 0

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"AMP enabled: {amp_enabled}"
    )

    print(
        f"DataLoader workers: "
        f"{num_workers}"
    )

    print(
        f"Pin memory: {pin_memory}"
    )

    # ========================================================
    # DATA
    # ========================================================

    train_loader, val_loader, _ = (
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

    # ========================================================
    # LOAD BEST FROZEN MODEL
    # ========================================================

    source_checkpoint = torch.load(
        SOURCE_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ]
    )

    # layer4 + fc become trainable.
    model = enable_partial_finetuning(
        model
    )

    model = model.to(device)

    # ========================================================
    # PARAMETER CHECK
    # ========================================================

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_params = (
        total_params
        - trainable_params
    )

    print()
    print("=" * 60)
    print(
        "PARTIAL FINE-TUNING CONFIGURATION"
    )
    print("=" * 60)

    print(
        f"Source checkpoint epoch: "
        f"{source_checkpoint['epoch']}"
    )

    print(
        f"Source validation accuracy: "
        f"{source_checkpoint['val_accuracy'] * 100:.2f}%"
    )

    print()
    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Frozen parameters: "
        f"{frozen_params:,}"
    )

    # ========================================================
    # SAFETY ASSERTIONS
    # ========================================================

    assert (
        model.conv1.weight.requires_grad
        is False
    )

    assert (
        model.layer3[0]
        .conv1.weight.requires_grad
        is False
    )

    assert (
        model.layer4[0]
        .conv1.weight.requires_grad
        is True
    )

    assert (
        model.fc.weight.requires_grad
        is True
    )

    # ========================================================
    # BATCHNORM SANITY CHECK
    # ========================================================

    model.train()
    set_frozen_batchnorm_eval(model)

    assert (
        model.bn1.training
        is False
    )

    assert (
        model.layer3[0]
        .bn1.training
        is False
    )

    assert (
        model.layer4[0]
        .bn1.training
        is True
    )

    assert (
        model.layer3[0]
        .bn1.weight.requires_grad
        is False
    )

    assert (
        model.layer4[0]
        .bn1.weight.requires_grad
        is True
    )

    print()
    print("Trainable layers:")
    print("  layer4: True")
    print("  fc: True")

    print()
    print("Frozen BatchNorm:")
    print("  bn1: True")
    print("  layer1: True")
    print("  layer2: True")
    print("  layer3: True")
    print("  layer4: False")

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # DIFFERENTIAL LEARNING RATES
    # ========================================================

    layer4_initial_lr = 1e-4
    fc_initial_lr = 5e-4

    optimizer = torch.optim.Adam(
        [
            {
                "params": (
                    model.layer4.parameters()
                ),
                "lr": layer4_initial_lr,
            },
            {
                "params": (
                    model.fc.parameters()
                ),
                "lr": fc_initial_lr,
            },
        ]
    )

    print()
    print(
        f"layer4 learning rate: "
        f"{layer4_initial_lr:.1e}"
    )

    print(
        f"fc learning rate: "
        f"{fc_initial_lr:.1e}"
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=[
                1e-6,
                5e-6,
            ],
        )
    )

    # ========================================================
    # HISTORY
    # ========================================================

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "layer4_learning_rate": [],
        "fc_learning_rate": [],
    }

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    early_stopping_counter = 0

    early_stopping_patience = 6

    min_delta = 1e-4

    num_epochs = 50

    print()
    print(
        f"Starting training for up to "
        f"{num_epochs} epochs"
    )

    print(
        f"Early stopping patience: "
        f"{early_stopping_patience}"
    )

    print(
        "Scheduler: ReduceLROnPlateau "
        "(patience=2, factor=0.5)"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(num_epochs):
        print()
        print("=" * 60)

        print(
            f"Epoch "
            f"{epoch + 1}/{num_epochs}"
        )

        print("=" * 60)

        epoch_layer4_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_fc_lr = (
            optimizer.param_groups[1]["lr"]
        )

        (
            train_loss,
            train_accuracy,
            train_time,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            non_blocking=non_blocking,
        )

        (
            val_loss,
            val_accuracy,
        ) = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            non_blocking=non_blocking,
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history["train_loss"].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        history[
            "layer4_learning_rate"
        ].append(
            epoch_layer4_lr
        )

        history[
            "fc_learning_rate"
        ].append(
            epoch_fc_lr
        )

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

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(val_loss)

        current_layer4_lr = (
            optimizer.param_groups[0]["lr"]
        )

        current_fc_lr = (
            optimizer.param_groups[1]["lr"]
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            val_loss
            < best_val_loss - min_delta
        ):
            best_val_loss = val_loss

            early_stopping_counter = 0

            print(
                f"Validation loss improved "
                f"to {best_val_loss:.4f}"
            )

        else:
            early_stopping_counter += 1

            print(
                f"No validation loss "
                f"improvement "
                f"({early_stopping_counter}/"
                f"{early_stopping_patience})"
            )

        # ====================================================
        # BEST MODEL BY VALIDATION ACCURACY
        # ====================================================

        if (
            val_accuracy
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_accuracy
            )

            output_checkpoint = {
                "experiment": (
                    EXPERIMENT_NAME
                ),
                "finetuning_mode": (
                    "partial"
                ),
                "trainable_layers": [
                    "layer4",
                    "fc",
                ],
                "source_experiment": (
                    "resnet18_frozen"
                ),
                "source_epoch": (
                    source_checkpoint[
                        "epoch"
                    ]
                ),
                "source_val_accuracy": (
                    source_checkpoint[
                        "val_accuracy"
                    ]
                ),
                "num_classes": 101,
                "amp_enabled": (
                    amp_enabled
                ),
                "epoch": epoch + 1,
                "layer4_learning_rate": (
                    epoch_layer4_lr
                ),
                "fc_learning_rate": (
                    epoch_fc_lr
                ),
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                ),
                "scaler_state_dict": (
                    scaler.state_dict()
                ),
                "train_loss": train_loss,
                "train_accuracy": (
                    train_accuracy
                ),
                "val_loss": val_loss,
                "val_accuracy": (
                    val_accuracy
                ),
            }

            torch.save(
                output_checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"Best model saved "
                f"(val acc: "
                f"{best_val_accuracy * 100:.2f}%)"
            )

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print()

        print(
            f"Epoch "
            f"{epoch + 1}/{num_epochs} "
            f"completed"
        )

        print(
            f"Train | "
            f"loss: {train_loss:.4f} | "
            f"acc: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val   | "
            f"loss: {val_loss:.4f} | "
            f"acc: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"LR layer4 | "
            f"{epoch_layer4_lr:.1e} "
            f"-> "
            f"{current_layer4_lr:.1e}"
        )

        print(
            f"LR fc     | "
            f"{epoch_fc_lr:.1e} "
            f"-> "
            f"{current_fc_lr:.1e}"
        )

        print(
            f"Train time: "
            f"{train_time:.1f} seconds"
        )

        # ====================================================
        # STOP
        # ====================================================

        if (
            early_stopping_counter
            >= early_stopping_patience
        ):
            print()

            print(
                f"Early stopping triggered "
                f"at epoch {epoch + 1}."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

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
        f"Frozen baseline accuracy: "
        f"{source_checkpoint['val_accuracy'] * 100:.2f}%"
    )

    improvement = (
        best_val_accuracy
        - source_checkpoint[
            "val_accuracy"
        ]
    ) * 100

    print(
        f"Improvement over frozen: "
        f"{improvement:+.2f} pp"
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