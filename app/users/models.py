from bson import ObjectId

def user_to_response(user_doc):
    return {
        "id": str(user_doc["_id"]),
        "email": user_doc["email"],
        "age": user_doc["age"]
    }