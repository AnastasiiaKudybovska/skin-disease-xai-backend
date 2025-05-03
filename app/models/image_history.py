from pydantic import BaseModel
from typing import List

class ImageHistory(BaseModel):
    image_url: str
    prediction: str
    confidence: float
    timestamp: str