import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from app.labels import FASHION_MNIST_LABELS
from app.model import FashionCNN
from config import BATCH_SIZE, MODEL_PATH, REPORTS_DIR, TEST_CSV


class FashionMNISTCSVDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.labels = dataframe["label"].astype("int64").values
        self.images = dataframe.drop(columns=["label"]).values.astype("float32") / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, 28, 28)
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.shape[1] == 785 and "label" not in df.columns:
        df = df.rename(columns={df.columns[0]: "label"})

    if "label" not in df.columns:
        df = pd.read_csv(csv_path, header=None)
        df = df.rename(columns={0: "label"})

    return df


def load_trained_model() -> tuple[FashionCNN, torch.device]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def get_predictions(model: FashionCNN, loader: DataLoader, device: torch.device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def save_classification_report(y_true, y_pred) -> None:
    target_names = [FASHION_MNIST_LABELS[i] for i in range(10)]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    output_path = REPORTS_DIR / "classification_report.csv"
    report_df.to_csv(output_path)

    print(f"Saved classification report to {output_path}")


def save_confusion_matrix(y_true, y_pred) -> None:
    target_names = [FASHION_MNIST_LABELS[i] for i in range(10)]
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix)

    ax.set_title("Fashion-MNIST Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)

    fig.colorbar(image)
    fig.tight_layout()

    output_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix to {output_path}")


def save_metrics_summary(y_true, y_pred) -> None:
    accuracy = accuracy_score(y_true, y_pred)

    summary = {
        "test_accuracy": round(float(accuracy), 4),
        "num_test_examples": int(len(y_true)),
        "model_path": str(MODEL_PATH),
    }

    output_path = REPORTS_DIR / "metrics_summary.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved metrics summary to {output_path}")
    print(json.dumps(summary, indent=2))


def evaluate() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = load_csv(TEST_CSV)
    test_dataset = FashionMNISTCSVDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model, device = load_trained_model()

    print(f"Using device: {device}")
    print(f"Evaluating model on {len(test_dataset)} test examples")

    y_true, y_pred = get_predictions(model, test_loader, device)

    save_classification_report(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred)
    save_metrics_summary(y_true, y_pred)


if __name__ == "__main__":
    evaluate()