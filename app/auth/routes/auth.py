from fastapi import APIRouter, HTTPException
from app.auth.hashing import hash_password, verify_password
from app.auth.schemas import UserCreate, UserLogin, UserResponse
from app.db.mongo import get_mongo_db
from datetime import datetime, date 

auth_router = APIRouter()

@auth_router.post("/signup", response_model=UserResponse)
def register_user(user: UserCreate):
    db = get_mongo_db()

    if db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_data = user.model_dump()
    user_data["password"] = hash_password(user.password)

    if isinstance(user_data["date_of_birth"], date):
        user_data["date_of_birth"] = datetime.combine(user_data["date_of_birth"], datetime.min.time())

    result = db.users.insert_one(user_data)

    user_data["id"] = str(result.inserted_id)
    del user_data["password"]
    return user_data


@auth_router.post("/signin", response_model=UserResponse)
def signin_user(login_data: UserLogin):
    db = get_mongo_db()
    user = db.users.find_one({"email": login_data.email})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid password")

    user["id"] = str(user["_id"])
    del user["_id"]
    del user["password"]

    return user