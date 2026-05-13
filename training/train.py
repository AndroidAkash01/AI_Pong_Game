from __future__ import annotations
import argparse
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from training.dataset import PongDataset
from training.model import PongMLP


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_data_file(side: str):
    if side == "left":
        return config.LEFT_DATA_FILE
    if side == "right":
        return config.RIGHT_DATA_FILE
    raise ValueError("side must be 'left' or 'right'")


def get_model_file(side: str):
    if side == "left":
        return config.LEFT_MODEL_FILE
    if side == "right":
        return config.RIGHT_MODEL_FILE
    raise ValueError("side must be 'left' or 'right'")


def train_model(
    side: str,
    hidden_sizes: list[int] | tuple[int, ...] = (15, 15),
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
) -> None:
    set_seed()

    data_file = get_data_file(side)
    model_file = get_model_file(side)

    dataset = PongDataset(data_file=data_file, side=side)
    total_size = len(dataset)

    labels = [y.item() for _, y in dataset]
    label_counts = Counter(labels)
    print("\nLabel distribution:", label_counts)

    total = sum(label_counts.values())
    class_weights = []
    for class_idx in range(3):
        count = label_counts.get(class_idx, 0)
        if count == 0:
            raise ValueError(f"Class {class_idx} has zero samples. Cannot train safely.")
        class_weights.append(total / (3 * count))

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights:", class_weights_tensor.tolist())

    val_size = max(1, int(0.2 * total_size))
    train_size = total_size - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_size = len(dataset[0][0])
    model = PongMLP(input_size=input_size, hidden_sizes=list(hidden_sizes))

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining model for side: {side}")
    print(f"Samples: {total_size}")
    print(f"Input size: {input_size}")
    print(f"Hidden layers: {list(hidden_sizes)}")
    print(f"Saving to: {model_file}\n")

    all_preds: list[int] = []
    all_labels: list[int] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)

                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    try:
        from sklearn.metrics import confusion_matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
    except Exception:
        print("\nInstall scikit-learn to print the confusion matrix:")
        print("pip install scikit-learn")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_sizes": list(hidden_sizes),
            "side": side,
        },
        model_file,
    )

    print(f"\nDone. Model saved to {model_file}\n")


def parse_hidden_sizes(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Provide hidden sizes like 15,15")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Hidden sizes must be integers like 15,15") from exc

    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Hidden sizes must be positive integers")

    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument(
        "--hidden_sizes",
        type=parse_hidden_sizes,
        default=[15, 15],
        help="Comma-separated hidden layer sizes, e.g. 15,15 or 30",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        side=args.side,
        hidden_sizes=args.hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()