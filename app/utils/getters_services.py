
from pymongo.database import Database
from bson import ObjectId

def get_user_by_email(db: Database, email: str):
    return db.users.find_one({"email": email})

def get_user_by_id(db: Database, id: str):
    object_id = ObjectId(id)
    return db.users.find_one({"_id": object_id})

def get_histories_by_user_id(db: Database, id: str):
    object_id = ObjectId(id)
    return db.histories.find({"user_id": object_id})