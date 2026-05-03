import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.model import FashionCNN
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, MODELS_DIR, TRAIN_CSV


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


def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    if df.shape[1] == 785 and "label" not in df.columns:
        df = df.rename(columns={df.columns[0]: "label"})

    if "label" not in df.columns:
        df = pd.read_csv(csv_path, header=None)
        df = df.rename(columns={0: "label"})

    return df


def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_csv(TRAIN_CSV)

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["label"],
    )

    train_dataset = FashionMNISTCSVDataset(train_df)
    val_dataset = FashionMNISTCSVDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = train_loss / total

        val_acc = evaluate_accuracy(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def evaluate_accuracy(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    train()