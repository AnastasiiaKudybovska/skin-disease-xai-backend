from fastapi import UploadFile
from app.models.model_loader import model
from app.services.preprocess import load_and_preprocess_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.xai.gradcam import generate_gradcam_for_image
import numpy as np

class_labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

async def explain_image_with_gradcam(file: UploadFile):
    # Зчитування і підготовка зображення
    image_data = await file.read()
    image_np = load_and_preprocess_image(image_data)  # повертає [224, 224, 3]
    image_np = preprocess_input(image_np)             # EfficientNet нормалізація

    # Отримання GradCAM
    pred_class, heatmap, overlay, masked_output, probs = generate_gradcam_for_image(
        image_np, model, layer_name="conv5_block3_3_conv"
    )

    return {
        "predicted_class": class_labels[int(pred_class)],
        "predicted_probs": probs[0].tolist(),
        # "heatmap": heatmap,
        "overlay": overlay,
        # "masked_output": masked_output,
    }
