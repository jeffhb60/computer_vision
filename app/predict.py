from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.labels import FASHION_MNIST_LABELS
from app.model import FashionCNN
from config import MODEL_PATH


_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


def load_model(model_path: Path = MODEL_PATH) -> FashionCNN:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = _transform(image)
    return tensor.unsqueeze(0)


def predict_image(model: FashionCNN, image: Image.Image) -> dict:
    device = next(model.parameters()).device

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probabilities, k=3)

    top_3 = [
        {
            "label": FASHION_MNIST_LABELS[int(class_id)],
            "class_id": int(class_id),
            "confidence": round(float(prob), 4),
        }
        for prob, class_id in zip(top_probs, top_indices)
    ]

    best = top_3[0]

    return {
        "predicted_label": best["label"],
        "predicted_class_id": best["class_id"],
        "confidence": best["confidence"],
        "top_3": top_3,
    }


def extract_embedding(model: FashionCNN, image: Image.Image) -> list[float]:
    device = next(model.parameters()).device

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        embedding = model.extract_embedding(tensor).squeeze(0)

    return embedding.cpu().numpy().tolist()