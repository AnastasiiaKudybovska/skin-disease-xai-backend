from fastapi import UploadFile, HTTPException
from app.schemas.classify import ClassificationResponse
from app.classification_models.model_loader import model
from app.services.preprocess import load_and_preprocess_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

class_labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'] 
MIN_CONFIDENCE_THRESHOLD = 0.5

async def classify_image(file: UploadFile) -> ClassificationResponse:
    image_np = load_and_preprocess_image(await file.read())
    image_np = np.expand_dims(image_np, axis=0)
    image_np = preprocess_input(image_np)
    
    predictions = model.predict(image_np)[0]
    pred_class_idx = np.argmax(predictions)
    confidence = float(predictions[pred_class_idx])
    probabilities = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
    
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to classify the image with sufficient confidence (confidence={confidence:.2f} ({class_labels[pred_class_idx]}))."
        )
    
    return ClassificationResponse(
        predicted_class=class_labels[pred_class_idx],
        confidence=confidence,
        probabilities=probabilities
    )
