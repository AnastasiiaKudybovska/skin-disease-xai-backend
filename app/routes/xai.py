from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.schemas.xai import GradCAMResponse
from app.services.xai_service import explain_image_with_gradcam
import numpy as np
import base64
import cv2

router = APIRouter()

def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")

@router.post("/gradcam", response_model=GradCAMResponse)
async def gradcam_explanation(file: UploadFile = File(...)):
    try:
        result = await explain_image_with_gradcam(file)

        return GradCAMResponse(
            predicted_class=result["predicted_class"],
            predicted_probs=result["predicted_probs"],
            # heatmap=encode_image_to_base64(result["heatmap"]),
            overlay=encode_image_to_base64(cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)),
            # masked_output=encode_image_to_base64(result["masked_output"]),
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
