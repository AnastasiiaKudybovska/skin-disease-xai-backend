from pydantic import BaseModel
from typing import Dict
class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
