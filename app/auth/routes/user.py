from fastapi import APIRouter, Depends
from app.auth.schemas import UserResponse, UserBase, UserUpdate
from app.auth.dependencies import get_current_user
from app.db.mongo import get_mongo_db
from pymongo.database import Database
from app.utils.exceptions import ( 
    user_not_found_exception,
    no_data_to_update_exception,
    no_changes_made_exception
)
from app.utils.getters_services import get_user_by_id
from bson import ObjectId
from datetime import datetime, timezone, time, date
import dateutil.parser

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


@user_router.put("/", response_model=UserBase)
async def update_user(
    update_data: UserUpdate,
    user: UserResponse = Depends(get_current_user),
    db: Database = Depends(get_mongo_db)
):
    print(update_data)
    update_values = update_data.dict(exclude_unset=True, exclude_none=True)
    
    if not update_values:
        raise no_data_to_update_exception

    if 'date_of_birth' in update_values:
        date_value = update_values['date_of_birth']
        
        if isinstance(date_value, str):
            parsed_date = dateutil.parser.isoparse(date_value)
            update_values['date_of_birth'] = parsed_date.replace(tzinfo=timezone.utc)
          
        
        elif isinstance(date_value, date):
            update_values['date_of_birth'] = datetime.combine(
                date_value, time.min
            ).replace(tzinfo=timezone.utc)
    
    update_values["updated_at"] = datetime.now(timezone.utc)
    
    result = db.users.update_one(
        {"_id": ObjectId(user["_id"])},
        {"$set": update_values}
    )

    if result.modified_count == 0:
        raise no_changes_made_exception

    updated_user = get_user_by_id(db, user["_id"])
    return UserBase(
        first_name=updated_user["first_name"],
        last_name=updated_user["last_name"],
        username=updated_user["username"],
        email=updated_user["email"],
        date_of_birth=updated_user["date_of_birth"]
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
