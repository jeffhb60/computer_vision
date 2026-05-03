from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from app.predict import load_model, predict_image
from app.quality import score_image_quality


app = FastAPI(title="Fashion Vision Auditor")

model = load_model()


@app.get("/")
def root():
    return {
        "message": "Fashion Vision Auditor API",
        "endpoints": ["/predict", "/quality", "/audit"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    result = predict_image(model, image)
    return result


@app.post("/quality")
async def quality(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    result = score_image_quality(image)

    return {
        "quality_score": result.quality_score,
        "blur_score": result.blur_score,
        "brightness_score": result.brightness_score,
        "contrast_score": result.contrast_score,
        "quality_label": result.quality_label,
        "warnings": result.warnings,
    }


@app.post("/audit")
async def audit(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    prediction = predict_image(model, image)
    quality_result = score_image_quality(image)

    if quality_result.quality_label in {"good", "acceptable"} and prediction["confidence"] >= 0.8:
        recommendation = "accept"
    elif quality_result.quality_label == "review":
        recommendation = "manual_review"
    else:
        recommendation = "reject_or_retake"

    return {
        "prediction": prediction,
        "quality": {
            "quality_score": quality_result.quality_score,
            "blur_score": quality_result.blur_score,
            "brightness_score": quality_result.brightness_score,
            "contrast_score": quality_result.contrast_score,
            "quality_label": quality_result.quality_label,
            "warnings": quality_result.warnings,
        },
        "recommendation": recommendation,
    }