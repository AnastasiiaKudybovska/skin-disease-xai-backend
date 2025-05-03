from pydantic import BaseModel
from typing import List

class User(BaseModel):
    user_id: str
    name: str
    email: str
    age: int
    image_history: List[dict] = [] 
    