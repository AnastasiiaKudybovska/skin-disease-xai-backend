from pydantic import BaseModel
from typing import Dict, Optional, List

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class ClassificationHistoryResponse(BaseModel):
    id: str
    image_id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str

class ClassificationWithHistoryResponse(ClassificationResponse):
    image_id: Optional[str] = None
    history_id: Optional[str] = None


class ClassificationDetailedHistoryResponse(ClassificationHistoryResponse):
    explanations: List[List]