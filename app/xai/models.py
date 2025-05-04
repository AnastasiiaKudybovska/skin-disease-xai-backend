from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ExplanationItem:
    method: str
    overlay_image_id: Optional[str]  # GridFS або base64

@dataclass
class Explanation:
    history_id: Optional[str] = None
    explanations: List[ExplanationItem] = None