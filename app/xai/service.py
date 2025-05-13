from fastapi import UploadFile
import numpy as np
import matplotlib.cm as cm
from app.classification_models.model_loader import model
from app.constants import CLASS_LABELS
from app.utils.preprocess_image import load_and_preprocess_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.utils.saving_images import encode_image_to_base64, save_image_to_gridfs
from app.xai.models import ExplanationItem
from app.xai.methods.gradcam import generate_gradcam_for_image
from app.xai.methods.lime import generate_lime_for_image, get_lime_heatmap, get_lime_overlay
from app.xai.methods.anchor import generate_anchor_for_image
from app.xai.methods.shap import generate_shap_for_image, get_shap_heatmap
from app.xai.methods.integrated_gradients import IntegratedGradVisualizer, generate_integrated_gradients_for_image
from app.utils.exceptions import (
    user_history_not_found_exception,
    invalid_image_id_exception
)
from bson import ObjectId
from dataclasses import asdict
from skimage.color import label2rgb


def delete_old_explanation_image(db, old_image_id):
    try:
        db.fs.files.delete_one({"_id": ObjectId(old_image_id)})
        db.fs.chunks.delete_many({"files_id": ObjectId(old_image_id)})
    except Exception:
        raise invalid_image_id_exception


def handle_authenticated_user(
    db, file, method_name: str, overlay_bgr, heatmap_bgr, history_id
) -> ExplanationItem:
    if not history_id:
        raise user_history_not_found_exception

    filename_overlay = f"{method_name}_overlay_{file.filename}_{history_id}"
    image_id_overlay = save_image_to_gridfs(db, overlay_bgr, filename_overlay)
    
    filename_heatmap = f"{method_name}_heatmap_{file.filename}_{history_id}"
    image_id_heatmap = save_image_to_gridfs(db, heatmap_bgr, filename_heatmap)
    
    explanation_item = ExplanationItem(
        method=method_name,
        overlay_image_id=str(image_id_overlay),
        heatmap_image_id=str(image_id_heatmap)
    )

    existing = db.explanations.find_one({
        "history_id": ObjectId(history_id),
        "explanations.method": method_name
    })

    if existing:
        item = next((e for e in existing["explanations"] if e["method"] == method_name), None)
        if item:
            old_image_id_overlay = item.get("overlay_image_id")
            if old_image_id_overlay:
                delete_old_explanation_image(db, old_image_id_overlay)

            old_image_id_heatmap = item.get("heatmap_image_id")
            if old_image_id_heatmap:
                delete_old_explanation_image(db, old_image_id_heatmap)


        db.explanations.update_one(
            {"history_id": ObjectId(history_id)},
            {"$set": {
                    "explanations.$[elem].overlay_image_id": image_id_overlay,
                    "explanations.$[elem].heatmap_image_id": image_id_heatmap
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


def handle_unknown_user(method_name: str, overlay_bgr, heatmap_bgr) -> ExplanationItem:
    overlay_base64 = encode_image_to_base64(overlay_bgr)
    heatmap_base64 = encode_image_to_base64(heatmap_bgr)
    return ExplanationItem(
        method=method_name,
        overlay_image_id=overlay_base64,
        heatmap_image_id=heatmap_base64
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
        "heatmap": heatmap,
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
    
    heatmap_norm = get_lime_heatmap(explanation, predicted_class_idx)
    colormap = cm.get_cmap('Spectral')
    heatmap_rgb = colormap(heatmap_norm)[..., :3]
    heatmap_rgb_uint8 = (heatmap_rgb * 255).astype(np.uint8)

    overlay = get_lime_overlay(explanation, predicted_class_idx)

    return {
        "predicted_class": CLASS_LABELS[int(predicted_class_idx)],
        "predicted_probs": probs[0].tolist(),
        "heatmap": heatmap_rgb_uint8,
        "overlay": overlay
    }


async def explain_image_with_anchor(file: UploadFile):
    image_data = await file.read()
    image_np = load_and_preprocess_image(image_data)
    image_np = preprocess_input(image_np)

    explanation, predicted_class_idx, probs = generate_anchor_for_image(
        image_np, model
    )
    
    if explanation is None:
        return None
    
    overlay = explanation.anchor.astype(np.uint8)
    heatmap = explanation.segments
    
    num_labels = int(heatmap.max()) + 1

    colormap = cm.get_cmap('Spectral', num_labels)
    colors = [colormap(i)[:3] for i in range(num_labels)]  
    
    heatmap_rgb = label2rgb(heatmap, colors=colors, bg_label=0)
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

    return {
        "predicted_class": CLASS_LABELS[int(predicted_class_idx)],
        "predicted_probs": probs[0].tolist(),
        "heatmap": heatmap_rgb,
        "overlay": overlay
    }


async def explain_image_with_shap(file: UploadFile):
    image_data = await file.read()
    image_np = load_and_preprocess_image(image_data)
    image_np = preprocess_input(image_np)

    explanation, predicted_class_idx, probs = generate_shap_for_image(
        image_np, model, top_k=1
    )
    
    if explanation is None:
        return None
    
    overlay = image_np
    heatmap = get_shap_heatmap(explanation)

    colormap = cm.get_cmap('Spectral')
    shap_rgb = colormap(heatmap)[..., :3]
    shap_rgb_uint8 = (shap_rgb * 255).astype(np.uint8)

    return {
        "predicted_class": CLASS_LABELS[int(predicted_class_idx)],
        "predicted_probs": probs[0].tolist(),
        "heatmap": shap_rgb_uint8,
        "overlay": overlay
    }


async def explain_image_with_integrated_gradients(file: UploadFile):
    image_data = await file.read()
    original_image = load_and_preprocess_image(image_data)  
    
    image_np = preprocess_input(original_image.astype(np.float32)) 

    grads, igrads, predicted_class_idx, probs = generate_integrated_gradients_for_image(image_np, model)

    if grads is None or igrads is None:
        return None

    vis = IntegratedGradVisualizer()

    igrads_attr, igrads_attr_outlines = vis.get_ig_attr_with_outlines(
        original_image.astype(np.float32),  
        gradients=grads.numpy(), 
        integrated_gradients=igrads.numpy(), 
        overlay=True
    )

    overlay = igrads_attr_outlines.astype(np.uint8)
    heatmap = igrads_attr.astype(np.uint8)

    return {
        "predicted_class": CLASS_LABELS[int(predicted_class_idx)],
        "predicted_probs": probs[0].tolist(),
        "heatmap": heatmap,
        "overlay": overlay
    }
