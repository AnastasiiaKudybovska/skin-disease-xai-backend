from fastapi import APIRouter, Depends, status
from app.auth.schemas import UserResponse, UserBase
from app.auth.service import get_current_user
from app.db.mongo import get_mongo_db
from pymongo.database import Database
from app.utils.exceptions import ( 
    user_not_found_exception
)
from bson import ObjectId

user_router = APIRouter()


@user_router.get("/", response_model=UserBase)
def get_profile(user: UserResponse = Depends(get_current_user)):
    return UserBase(
        first_name=user["first_name"],
        last_name=user["last_name"],
        username=user["username"],
        email=user["email"],
        date_of_birth=user["date_of_birth"]
    ) 


@user_router.delete("/")
def delete_user(
    user: UserResponse = Depends(get_current_user),
    db: Database = Depends(get_mongo_db)
):
    result = db.users.delete_one({"_id": ObjectId(user["_id"])})
    if result.deleted_count == 0:
        raise user_not_found_exception
    return {"message": "User deleted"}
