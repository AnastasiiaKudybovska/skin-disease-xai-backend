from fastapi import APIRouter, UploadFile, File, Depends, Form
from fastapi.responses import JSONResponse
from app.auth.dependencies import get_current_user_optional
from app.db.mongo import get_mongo_db
from app.utils.saving_images import get_image_from_gridfs, save_image_to_gridfs, encode_image_to_base64
from app.utils.exceptions import user_history_not_found_exception, invalid_image_id_exception, image_not_found_exception
from app.xai.models import Explanation
from app.xai.schemas import GradCAMResponse
from app.xai.service import explain_image_with_gradcam
import cv2
from typing import Optional
from pymongo.database import Database
from bson import ObjectId
from dataclasses import asdict

xai_router = APIRouter()


@xai_router.post("/gradcam", response_model=GradCAMResponse)
async def gradcam_explanation(
    file: UploadFile = File(...),
    history_id: Optional[str] = Form(None),
    db: Database = Depends(get_mongo_db),
    user: Optional[dict] = Depends(get_current_user_optional)
):
    result = await explain_image_with_gradcam(file)

    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    image_id = None
    overlay_image = None
        
    if user:
        filename = f"gradcam_{file.filename}_{history_id}"
        image_id = save_image_to_gridfs(db, overlay_bgr, filename)
        explanation_doc = {
            "method": "gradcam",
            "overlay_image_id": image_id,
        }
        if history_id:
            explanation_doc["history_id"] = ObjectId(history_id)
        else:
            raise user_history_not_found_exception

        db.explanations.insert_one(explanation_doc)
        overlay_image = image_id
    else:
        overlay_base64 = encode_image_to_base64(result["overlay"]) 
        overlay_image = overlay_base64

    explanation = Explanation(
        method="gradcam",
        overlay_image_id=overlay_image,
        history_id=history_id if history_id else None
    )
    
    return GradCAMResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=[asdict(explanation)],
    )


@xai_router.get("/get_image/{image_id}")
async def get_image(image_id: str, db: Database = Depends(get_mongo_db)):
    try:
        object_id = ObjectId(image_id)
    except Exception:
        raise invalid_image_id_exception
    
    try:
        return get_image_from_gridfs(db, image_id)
    except Exception:
        raise image_not_found_exception
