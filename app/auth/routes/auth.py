from fastapi import APIRouter, Depends
from app.auth.hashing import hash_password, verify_password
from app.auth.models import User
from app.auth.schemas import UserCreate, UserLogin, UserResponse, UserLoginResponse
from app.auth.dependencies import get_current_user
from app.auth.service import update_user_tokens
from app.db.mongo import get_mongo_db
from bson import ObjectId

from app.utils.getters_services import get_user_by_email, get_user_by_id 
from app.utils.exceptions import ( 
    email_already_registered_exception, 
    email_not_registered_exception, 
    invalid_credentials_exception,
    invalid_token_exception,
    user_not_found_exception
)
from app.utils.jwt_handlers import create_access_token, create_refresh_token

auth_router = APIRouter()

@auth_router.post("/signup", response_model=UserResponse)
def register_user(user: UserCreate):
    db = get_mongo_db()
    
    existing_user = get_user_by_email(db, user.email)
    if existing_user:
        raise email_already_registered_exception

    user_data = User(
        first_name=user.first_name,
        last_name=user.last_name,
        username=user.username,
        email=user.email,
        date_of_birth=user.date_of_birth,
        password=hash_password(user.password)
    )

    db_user = user_data.to_dict()
    result = db.users.insert_one(db_user)

    user_data_dict = user_data.to_dict()
    user_data_dict["id"] = str(result.inserted_id)
    del user_data_dict["password"]

    return user_data_dict


@auth_router.post("/signin", response_model=UserLoginResponse)
def sign_in(user: UserLogin):
    db = get_mongo_db()

    existing_user = get_user_by_email(db, user.email)
    if not existing_user:
        raise email_not_registered_exception
    
    if not verify_password(user.password, existing_user["password"]):
        raise invalid_credentials_exception


    user_id = str(existing_user["_id"])

    access_token = create_access_token(data={"sub": user_id})
    refresh_token = create_refresh_token(data={"sub": user_id})
    user_id = str(existing_user["_id"])
    update_user_tokens(db, user_id, access_token, refresh_token)

    return UserLoginResponse(
        first_name=existing_user["first_name"],
        last_name=existing_user["last_name"],
        username=existing_user["username"],
        email=existing_user["email"],
        date_of_birth=existing_user["date_of_birth"],
        access_token=access_token,
        refresh_token=refresh_token
    )


@auth_router.patch("/logout")
async def logout(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_mongo_db)
):
    try:
        db.users.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": {"access_token": None, "refresh_token": None}}
        )
        return {"message": "User logged out successfully"}
    except Exception as e:
        raise invalid_token_exception
    