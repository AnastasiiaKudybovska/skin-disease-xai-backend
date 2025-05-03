from fastapi import UploadFile
from app.schemas.classify import ClassificationResponse
import io
from PIL import Image
import base64

class DummyModel:
    def predict(self, image):
        return "melanoma", 0.92

model = DummyModel()

async def classify_image(file: UploadFile) -> ClassificationResponse:
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    label, confidence = model.predict(image)

    return ClassificationResponse(
        predicted_class=label,
        confidence=confidence
    )
