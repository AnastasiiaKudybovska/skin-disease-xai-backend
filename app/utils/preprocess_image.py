import numpy as np
import cv2
from PIL import Image
import io

def clean_skin_image(image_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, binary_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)
    clean_image = cv2.inpaint(image_np, binary_mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
    return clean_image


def load_and_preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image)
    image_np = clean_skin_image(image_np)
    return image_np

def load_and_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    return image_np