from dataclasses import dataclass
from typing import Dict

@dataclass
class Histrory:
    image_id: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str