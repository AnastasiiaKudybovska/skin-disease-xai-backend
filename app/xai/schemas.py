from pydantic import BaseModel
from typing import List, Optional
from bson import ObjectId

class GradCAMResponse(BaseModel):
    predicted_class: str
    predicted_probs: List[float]
    explanations: List[dict]