import numpy as np
import cv2
import base64
from io import BytesIO
from fastapi.responses import StreamingResponse
from bson import ObjectId
import gridfs

def save_image_to_gridfs(db, image, filename):
    fs = gridfs.GridFS(db)
    _, buffer = cv2.imencode(".png", image)
    file_id = fs.put(buffer.tobytes(), filename=filename)
    return str(file_id)


def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def get_image_from_gridfs(db, image_id):
    fs = gridfs.GridFS(db)
    file_data = fs.get(ObjectId(image_id)).read()
    return StreamingResponse(BytesIO(file_data), media_type="image/jpeg")
