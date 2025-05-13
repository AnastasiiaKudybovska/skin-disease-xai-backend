from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import contextlib
import io


def generate_lime_for_image(image, model, top_labels=5, num_samples=300): # 1000
    def silent_predict(images):
        with contextlib.redirect_stdout(io.StringIO()):
            return model.predict(images, verbose=0)
    
    image_batch = np.expand_dims(image, axis=0)
    
    preds = model.predict(image_batch, verbose=0)
    predicted_class_idx = np.argmax(preds[0])

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, silent_predict, top_labels=top_labels, num_samples=num_samples, hide_color=0)

    if predicted_class_idx not in explanation.local_exp:
        # print(f"Class {predicted_class_idx} not found in explanation, skipping...")
        return None, None, None

    return predicted_class_idx, explanation, preds


def get_lime_heatmap(explanation, predicted_class_idx):
    dict_heatmap = dict(explanation.local_exp[predicted_class_idx])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap_norm


def get_lime_overlay(explanation, predicted_class_idx, num_features=10):
    temp, mask = explanation.get_image_and_mask(
        predicted_class_idx,
        positive_only=False,
        hide_rest=False,
        num_features=num_features
    )
    overlay = mark_boundaries(temp / 255.0, mask)
    return (overlay * 255).astype("uint8")
