from fastapi import APIRouter, UploadFile, File, Depends
from app.classification.service import classify_image
from app.classification.schemas import ClassificationHistoryResponse, ClassificationWithHistoryResponse
from app.db.mongo import get_mongo_db
from app.auth.dependencies import get_current_user, get_current_user_optional
from pymongo.database import Database
from typing import Optional, List
from bson import ObjectId
from app.utils.getters_services import get_histories_by_user_id
from app.utils.exceptions import user_history_not_found_exception, invalid_history_id_exception
from app.utils.history_cleanup import delete_history_with_related

classify_router = APIRouter()


@classify_router.post("/", response_model=ClassificationWithHistoryResponse)
async def skin_classification(
    file: UploadFile = File(...),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    return await classify_image(file, db=db, user=user)


@classify_router.get("/histories", response_model=List[ClassificationHistoryResponse])
async def get_user_history(
    db: Database = Depends(get_mongo_db),
    user: dict = Depends(get_current_user)
):
    histories  = get_histories_by_user_id(db, user["_id"])
    return [
        ClassificationHistoryResponse(
            id=str(item["_id"]),
            predicted_class=item["predicted_class"],
            confidence=item["confidence"],
            probabilities=item["probabilities"],
            timestamp=item["timestamp"]
        )
        for item in histories
    ]


@classify_router.delete("/histories/{history_id}")
async def delete_history(
    history_id: str,
    db: Database = Depends(get_mongo_db),
    user: dict = Depends(get_current_user)
):
    try:
        object_id = ObjectId(history_id)
    except Exception:
        raise invalid_history_id_exception

    history = db.histories.find_one({
        "_id": object_id,
        "user_id": ObjectId(user["_id"])
    })

    if not history:
        raise user_history_not_found_exception

    delete_history_with_related(db, object_id)
    return {"message": "User history and related images deleted"}
    

@classify_router.delete("/histories")
async def delete_all_histories(
    db: Database = Depends(get_mongo_db),
    user: dict = Depends(get_current_user)
):
    user_id = ObjectId(user["_id"])
    histories = db.histories.find({"user_id": user_id})
    
    for history in histories:
        delete_history_with_related(db, history["_id"])

    return {"message": "All user histories and related explanations/images deleted"}