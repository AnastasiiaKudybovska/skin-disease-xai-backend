from fastapi import APIRouter, UploadFile, File, Depends
from app.classification.service import classify_image
from app.classification.schemas import ClassificationResponse
from app.db.mongo import get_mongo_db
from app.auth.dependencies import get_current_user_optional
from pymongo.database import Database
from typing import Optional

classify_router = APIRouter()

@classify_router.post("/", response_model=ClassificationResponse)
async def predict(
    file: UploadFile = File(...),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    return await classify_image(file, db=db, user=user)