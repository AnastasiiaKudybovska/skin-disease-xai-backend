from pymongo import MongoClient
from app.db.config import MONGODB_URL, MONGO_DB_NAME

client = MongoClient(MONGODB_URL)
db = client[MONGO_DB_NAME]

def get_mongo_db():
    return db