from pydantic import BaseModel
from typing import Dict, Optional

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

class ClassificationWithHistoryResponse(ClassificationResponse):
    history_id: Optional[str] = None
