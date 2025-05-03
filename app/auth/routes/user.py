from fastapi import APIRouter, Depends, HTTPException
from app.auth.schemas import UserResponse
from app.db.mongo import get_mongo_db

user_router = APIRouter()

@user_router.get("/{email}", response_model=UserResponse)
def get_user(email: str):
    db = get_mongo_db()
    user_doc = db.users.find_one({"email": email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": str(user_doc["_id"]),
        "email": user_doc["email"],
        "age": user_doc["age"]
    }

@user_router.delete("/{email}")
def delete_user(email: str):
    db = get_mongo_db()
    result = db.users.delete_one({"email": email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"msg": "User deleted"}