from gridfs import GridFS
from bson import ObjectId
from pymongo.database import Database
from app.utils.exceptions import invalid_image_id_exception

def delete_history_with_related(db: Database, history_id: ObjectId):
    explanations = list(db.explanations.find({"history_id": history_id}))
    fs = GridFS(db)

    for explanation in explanations:
        image_id = explanation.get("overlay_image_id")
        if image_id:
            try:
                fs.delete(ObjectId(image_id))
            except Exception as e:
                raise invalid_image_id_exception

        db.explanations.delete_one({"_id": explanation["_id"]})

    db.histories.delete_one({"_id": history_id})
