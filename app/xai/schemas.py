from pydantic import BaseModel
from typing import List

class GradCAMResponse(BaseModel):
    predicted_class: str
    predicted_probs: List[float]
    # heatmap: str  # base64
    overlay: str  # base64
    # masked_output: str  # base64
