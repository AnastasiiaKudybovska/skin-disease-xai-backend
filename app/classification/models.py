from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Histrory:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str