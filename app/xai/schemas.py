from pydantic import BaseModel
from typing import List, Optional
from bson import ObjectId

class XAIResponse(BaseModel):
    predicted_class: str
    predicted_probs: List[float]
    explanations: List[dict]