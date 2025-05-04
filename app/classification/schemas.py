from pydantic import BaseModel, Field
from typing import Dict

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class ClassificationHistoryResponse(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str