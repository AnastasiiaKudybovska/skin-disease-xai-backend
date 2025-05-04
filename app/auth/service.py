from datetime import datetime, timezone
from bson import ObjectId
from fastapi import Depends
from app.db.mongo import get_mongo_db
from app.utils.getters_services import get_user_by_id
from app.utils.jwt_handlers import decode_access_token
from pymongo.database import Database
from typing import Optional

from app.utils.exceptions import ( 
    invalid_token_exception,
    user_not_found_exception
)

def update_user_tokens(db, user_id: str, access_token: str, refresh_token: str):
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "updated_at": datetime.now(timezone.utc)
        }}
    )


def get_current_user(token: dict = Depends(decode_access_token), db: Database = Depends(get_mongo_db)):
    if not token:
        raise invalid_token_exception
    user_id = token.get("sub")
    user = get_user_by_id(db, user_id)
    if not user:
        raise user_not_found_exception
    return user 


# def get_current_user_optional(
#     token: Optional[dict] = Depends(decode_access_token),
#     db: Database = Depends(get_mongo_db)
# ) -> Optional[dict]:
#     if not token:
#         return None
#     user_id = token.get("sub")
#     if not user_id:
#         return None
#     user = get_user_by_id(db, user_id)
#     return user