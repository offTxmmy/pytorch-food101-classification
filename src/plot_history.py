import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

V1_HISTORY_PATH = (
    EXPERIMENTS_DIR / "foodcnn_v1" / "history.json"
)

V2_HISTORY_PATH = (
    EXPERIMENTS_DIR / "foodcnn_v2" / "history.json"
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_history(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def plot_accuracy(v1_history, v2_history):
    v1_epochs = range(
        1,
        len(v1_history["train_accuracy"]) + 1,
    )

    v2_epochs = range(
        1,
        len(v2_history["train_accuracy"]) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        v1_epochs,
        [
            accuracy * 100
            for accuracy in v1_history["train_accuracy"]
        ],
        label="FoodCNN V1 - Train",
    )

    plt.plot(
        v1_epochs,
        [
            accuracy * 100
            for accuracy in v1_history["val_accuracy"]
        ],
        label="FoodCNN V1 - Validation",
    )

    plt.plot(
        v2_epochs,
        [
            accuracy * 100
            for accuracy in v2_history["train_accuracy"]
        ],
        label="FoodCNN V2 - Train",
    )

    plt.plot(
        v2_epochs,
        [
            accuracy * 100
            for accuracy in v2_history["val_accuracy"]
        ],
        label="FoodCNN V2 - Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("FoodCNN V1 vs V2 - Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / "accuracy_comparison.png"

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Accuracy plot saved to: {output_path}")


def plot_loss(v1_history, v2_history):
    v1_epochs = range(
        1,
        len(v1_history["train_loss"]) + 1,
    )

    v2_epochs = range(
        1,
        len(v2_history["train_loss"]) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        v1_epochs,
        v1_history["train_loss"],
        label="FoodCNN V1 - Train",
    )

    plt.plot(
        v1_epochs,
        v1_history["val_loss"],
        label="FoodCNN V1 - Validation",
    )

    plt.plot(
        v2_epochs,
        v2_history["train_loss"],
        label="FoodCNN V2 - Train",
    )

    plt.plot(
        v2_epochs,
        v2_history["val_loss"],
        label="FoodCNN V2 - Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("FoodCNN V1 vs V2 - Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / "loss_comparison.png"

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Loss plot saved to: {output_path}")


def main():
    v1_history = load_history(V1_HISTORY_PATH)
    v2_history = load_history(V2_HISTORY_PATH)

    plot_accuracy(v1_history, v2_history)
    plot_loss(v1_history, v2_history)


if __name__ == "__main__":
    main()