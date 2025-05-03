from pydantic import BaseModel

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
