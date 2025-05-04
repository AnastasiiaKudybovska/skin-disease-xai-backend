

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.auth.service import decode_access_token
from app.db.mongo import get_mongo_db
from pymongo.database import Database
from app.utils.exceptions import user_not_found_exception, invalid_token_exception
from app.utils.getters_services import get_user_by_id

security_opt = HTTPBearer(auto_error=False)
security = HTTPBearer()

async def get_current_user_optional(cred: HTTPAuthorizationCredentials = Depends(security_opt), db: Database = Depends(get_mongo_db)):
    token = cred.credentials if cred else None # remove if cred else None if not optional
    if not token:
        return None
    
    try:
        decoded_token = decode_access_token(token)
        user_id = decoded_token.get("sub")
        user = get_user_by_id(db, user_id)
        if not user:
            raise user_not_found_exception
        return user
    except Exception:
        raise invalid_token_exception
    

async def get_current_user(cred: HTTPAuthorizationCredentials = Depends(security), db: Database = Depends(get_mongo_db)):
    token = cred.credentials
    if not token:
        raise invalid_token_exception
    try:
        decoded_token = decode_access_token(token)
        user_id = decoded_token.get("sub")
        user = get_user_by_id(db, user_id)
        if not user:
            raise user_not_found_exception
        return user
    except Exception:
        raise invalid_token_exception