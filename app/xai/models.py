from dataclasses import dataclass
from typing import Optional

@dataclass
class Explanation:
    method: str
    overlay_image_id: Optional[str]  # GridFS або base64
    history_id: Optional[str] = None
