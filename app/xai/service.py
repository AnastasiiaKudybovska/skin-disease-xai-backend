from fastapi import UploadFile
from app.classification_models.model_loader import model
from app.constants import CLASS_LABELS
from app.utils.preprocess_image import load_and_preprocess_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.utils.saving_images import encode_image_to_base64, save_image_to_gridfs
from app.xai.models import ExplanationItem
from app.xai.methods.gradcam import generate_gradcam_for_image
from app.xai.methods.lime import generate_lime_for_image, get_lime_overlay
from app.utils.exceptions import (
    user_history_not_found_exception,
    invalid_image_id_exception
)
from bson import ObjectId
from dataclasses import asdict


def delete_old_explanation_image(db, old_image_id):
    try:
        db.fs.files.delete_one({"_id": ObjectId(old_image_id)})
        db.fs.chunks.delete_many({"files_id": ObjectId(old_image_id)})
    except Exception:
        raise invalid_image_id_exception


def handle_authenticated_user(
    db, file, method_name: str, overlay_bgr, history_id
) -> ExplanationItem:
    if not history_id:
        raise user_history_not_found_exception

    filename = f"{method_name}_{file.filename}_{history_id}"
    image_id = save_image_to_gridfs(db, overlay_bgr, filename)
    
    explanation_item = ExplanationItem(
        method=method_name,
        overlay_image_id=str(image_id)
    )

    existing = db.explanations.find_one({
        "history_id": ObjectId(history_id),
        "explanations.method": method_name
    })

    if existing:
        item = next((e for e in existing["explanations"] if e["method"] == method_name), None)
        if item:
            old_image_id = item.get("overlay_image_id")
            if old_image_id:
                delete_old_explanation_image(db, old_image_id)

        db.explanations.update_one(
            {"history_id": ObjectId(history_id)},
            {"$set": {
                    "explanations.$[elem].overlay_image_id": image_id
                }
            },
            array_filters=[{"elem.method": method_name}]
        )
    else:
        db.explanations.update_one(
            {"history_id": ObjectId(history_id)},
            {"$push": {"explanations": asdict(explanation_item)}},
            upsert=True
        )

    return explanation_item


def handle_unknown_user(method_name: str, overlay_bgr) -> ExplanationItem:
    overlay_base64 = encode_image_to_base64(overlay_bgr)
    return ExplanationItem(
        method=method_name,
        overlay_image_id=overlay_base64
    )


async def explain_image_with_gradcam(file: UploadFile):
    # Зчитування і підготовка зображення
    image_data = await file.read()
    image_np = load_and_preprocess_image(image_data)
    image_np = preprocess_input(image_np)
    # Отримання GradCAM
    pred_class, heatmap, overlay, masked_output, probs = generate_gradcam_for_image(
        image_np, model, layer_name="conv5_block3_3_conv"
    )

    return {
        "predicted_class": CLASS_LABELS[int(pred_class)],
        "predicted_probs": probs[0].tolist(),
        # "heatmap": heatmap,
        "overlay": overlay,
        # "masked_output": masked_output,
    }


async def explain_image_with_lime(file: UploadFile):
    image_data = await file.read()
    image_np = load_and_preprocess_image(image_data)
    image_np = preprocess_input(image_np)

    predicted_class_idx, explanation, probs = generate_lime_for_image(
        image_np, model
    )
    
    if explanation is None:
        return None
    
    # heatmap = get_lime_heatmap(explanation, predicted_class_idx)
    overlay = get_lime_overlay(explanation, predicted_class_idx)


    return {
        "predicted_class": CLASS_LABELS[int(predicted_class_idx)],
        "predicted_probs": probs[0].tolist(),
        # "heatmap": heatmap,
        "overlay": overlay
    }

