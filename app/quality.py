from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageQualityResult:
    quality_score: float
    blur_score: float
    brightness_score: float
    contrast_score: float
    quality_label: str
    warnings: list[str]


def pil_to_grayscale_array(image: Image.Image) -> np.ndarray:
    image = image.convert("L").resize((28, 28))
    return np.array(image)


def score_blur(image_array: np.ndarray) -> float:
    laplacian_var = cv2.Laplacian(image_array, cv2.CV_64F).var()
    score = min(100.0, laplacian_var / 10.0)
    return float(score)


def score_brightness(image_array: np.ndarray) -> float:
    mean_brightness = image_array.mean()

    # Best range is roughly middle brightness.
    distance_from_mid = abs(mean_brightness - 127.5)
    score = max(0.0, 100.0 - (distance_from_mid / 127.5) * 100.0)

    return float(score)


def score_contrast(image_array: np.ndarray) -> float:
    contrast = image_array.std()
    score = min(100.0, contrast * 2.0)
    return float(score)


def score_image_quality(image: Image.Image) -> ImageQualityResult:
    image_array = pil_to_grayscale_array(image)

    blur = score_blur(image_array)
    brightness = score_brightness(image_array)
    contrast = score_contrast(image_array)

    quality_score = (
        0.40 * blur
        + 0.30 * brightness
        + 0.30 * contrast
    )

    warnings = []

    if blur < 35:
        warnings.append("Image appears blurry")

    if brightness < 35:
        warnings.append("Image brightness may be poor")

    if contrast < 35:
        warnings.append("Image has low contrast")

    if quality_score >= 85:
        quality_label = "good"
    elif quality_score >= 70:
        quality_label = "acceptable"
    elif quality_score >= 50:
        quality_label = "review"
    else:
        quality_label = "reject"

    return ImageQualityResult(
        quality_score=round(float(quality_score), 2),
        blur_score=round(float(blur), 2),
        brightness_score=round(float(brightness), 2),
        contrast_score=round(float(contrast), 2),
        quality_label=quality_label,
        warnings=warnings,
    )