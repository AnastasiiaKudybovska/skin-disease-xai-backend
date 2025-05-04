from fastapi import UploadFile, HTTPException, status
from app.classification.schemas import ClassificationResponse, ClassificationWithHistoryResponse
from app.classification_models.model_loader import model
from app.constants import CLASS_LABELS, MIN_CONFIDENCE_THRESHOLD
from app.utils.preprocess_image import load_and_preprocess_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from pymongo.database import Database
from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional

async def classify_image(
    file: UploadFile,
    db: Database,
    user: Optional[dict] = None
) -> ClassificationWithHistoryResponse:
    image_np = load_and_preprocess_image(await file.read())
    image_np = np.expand_dims(image_np, axis=0)
    image_np = preprocess_input(image_np)
    
    predictions = model.predict(image_np)[0]
    pred_class_idx = np.argmax(predictions)
    confidence = float(predictions[pred_class_idx])
    probabilities = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
    
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to classify the image with sufficient confidence (confidence={confidence:.2f} ({CLASS_LABELS[pred_class_idx]}))."
        )
    
    result = ClassificationResponse(
        predicted_class=CLASS_LABELS[pred_class_idx],
        confidence=confidence,
        probabilities=probabilities
        
    )
    
    history_id = None
    if user:
        inserted = db.histories.insert_one({
            "user_id": ObjectId(user["_id"]),
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        history_id = str(inserted.inserted_id)

    return ClassificationWithHistoryResponse(**result.model_dump(), history_id=history_id)
