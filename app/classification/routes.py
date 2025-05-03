from fastapi import APIRouter, UploadFile, File
from app.classification.service import classify_image
from app.classification.schemas import ClassificationResponse

classify_router = APIRouter()

@classify_router.post("/predict", response_model=ClassificationResponse)
async def predict(file: UploadFile = File(...)):
    return await classify_image(file)
