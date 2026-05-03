from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.labels import SAFE_LABEL_NAMES
from config import TRAIN_CSV, SAMPLE_IMAGES_DIR


def load_fashion_mnist_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.shape[1] == 785:
        if "label" not in df.columns:
            df = df.rename(columns={df.columns[0]: "label"})
        return df

    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] != 785:
        raise ValueError(
            f"Expected 785 columns: 1 label + 784 pixels. Got {df.shape[1]} columns."
        )

    df = df.rename(columns={0: "label"})
    return df


def export_images(csv_path: Path, output_dir: Path, limit: int = 1000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_fashion_mnist_csv(csv_path).head(limit)

    for idx, row in df.iterrows():
        label = int(row["label"])
        label_name = SAFE_LABEL_NAMES[label]

        pixels = row.drop("label").to_numpy(dtype=np.uint8)

        if pixels.size != 784:
            raise ValueError(f"Expected 784 pixels, got {pixels.size} at row {idx}")

        image_array = pixels.reshape(28, 28)

        class_dir = output_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        image = Image.fromarray(image_array, mode="L")
        image = image.resize((280, 280))
        image.save(class_dir / f"{idx:06d}_{label_name}.png")

    print(f"Exported {len(df)} images to {output_dir}")


if __name__ == "__main__":
    export_images(TRAIN_CSV, SAMPLE_IMAGES_DIR, limit=1000)