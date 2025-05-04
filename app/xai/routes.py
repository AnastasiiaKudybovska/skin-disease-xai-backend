from fastapi import APIRouter, UploadFile, File, Depends, Form
from fastapi.responses import JSONResponse
from gridfs import GridFS
from app.auth.dependencies import get_current_user_optional
from app.db.mongo import get_mongo_db
from app.utils.history_cleanup import delete_all_images
from app.utils.saving_images import get_image_from_gridfs, save_image_to_gridfs, encode_image_to_base64
from app.utils.exceptions import user_history_not_found_exception, invalid_image_id_exception, image_not_found_exception
from app.xai.models import Explanation, ExplanationItem
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
    explanations = []

    if user:
        if not history_id:
            raise user_history_not_found_exception

        filename = f"gradcam_{file.filename}_{history_id}"
        image_id = save_image_to_gridfs(db, overlay_bgr, filename)
        
        explanation_item = ExplanationItem(
            method="gradcam",
            overlay_image_id=str(image_id)
        )

        existing = db.explanations.find_one({
            "history_id": ObjectId(history_id),
            "explanations.method": "gradcam"
        })
        if existing:
            gradcam_item = next(
                (e for e in existing["explanations"] if e["method"] == "gradcam"),
                 None
            )     
            if gradcam_item:
                old_image_id = gradcam_item.get("overlay_image_id")
                if old_image_id:
                    try:
                        db.fs.files.delete_one({"_id": ObjectId(old_image_id)})
                        db.fs.chunks.delete_many({"files_id": ObjectId(old_image_id)})
                    except Exception as e:
                        raise invalid_image_id_exception

            db.explanations.update_one(
                {"history_id": ObjectId(history_id)},
                {
                    "$set": {
                        "explanations.$[elem].overlay_image_id": image_id
                    }
                },
                array_filters=[{"elem.method": "gradcam"}]
            )
        else:
            db.explanations.update_one(
                {"history_id": ObjectId(history_id)},
                {"$push": {"explanations": asdict(explanation_item)}},
                upsert=True
            )
        
        explanations.append(explanation_item)
    else:
        overlay_base64 = encode_image_to_base64(result["overlay"]) 

        explanation_item = ExplanationItem(
            method="gradcam",
            overlay_image_id=overlay_base64
        )
        explanations.append(explanation_item)

    explanation_response = Explanation(
        history_id=history_id,
        explanations=explanations
    )
    return GradCAMResponse(
        predicted_class=result["predicted_class"],
        predicted_probs=result["predicted_probs"],
        explanations=[asdict(explanation_response)]
    )


@xai_router.get("/images/{image_id}")
async def get_image(image_id: str, db: Database = Depends(get_mongo_db)):
    try:
        object_id = ObjectId(image_id)
    except Exception:
        raise invalid_image_id_exception
    
    try:
        return get_image_from_gridfs(db, image_id)
    except Exception:
        raise image_not_found_exception
    

@xai_router.get("/images/all_images")
async def list_all_images(db: Database = Depends(get_mongo_db)):
    fs = GridFS(db)
    files = fs.find()
    result = []

    for file in files:
        result.append({
            "filename": file.filename,
            "file_id": str(file._id),
            "upload_date": file.upload_date.isoformat() 
        })

    return JSONResponse(content=result)


@xai_router.delete("/images/delete_all_images")
async def delete_all_images_endpoint(db: Database = Depends(get_mongo_db)):
    try:
        delete_all_images(db)
        return JSONResponse(content={"message": "All images have been deleted successfully."})
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)