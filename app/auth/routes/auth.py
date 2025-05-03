from fastapi import APIRouter, HTTPException
from app.auth.hashing import hash_password, verify_password
from app.auth.schemas import UserCreate, UserLogin, UserResponse
from app.db.mongo import get_mongo_db
from app.users.models import user_to_response

auth_router = APIRouter()

@auth_router.post("/signup", response_model=UserResponse)
def signup(user: UserCreate):
    db = get_mongo_db()
    existing = db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = hash_password(user.password)
    new_user = {"email": user.email, "password": hashed_pw, "age": user.age}
    result = db.users.insert_one(new_user)
    new_user["_id"] = result.inserted_id
    return user_to_response(new_user)

@auth_router.post("/signin", response_model=UserResponse)
def signin(user: UserLogin):
    db = get_mongo_db()
    user_doc = db.users.find_one({"email": user.email})
    if not user_doc or not verify_password(user.password, user_doc["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user_to_response(user_doc)