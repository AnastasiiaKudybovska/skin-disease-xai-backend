from fastapi import APIRouter, UploadFile, File
from app.services.classifier_service import classify_image
from app.schemas.classify import ClassificationResponse

router = APIRouter()

@router.post("/predict", response_model=ClassificationResponse)
async def predict(file: UploadFile = File(...)):
    return await classify_image(file)
