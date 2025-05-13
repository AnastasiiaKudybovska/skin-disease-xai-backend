import cv2
import numpy as np
import shap

from app.constants import CLASS_LABELS

def generate_shap_for_image(image, model, class_labels=CLASS_LABELS, top_k=None, specific_classes=None):
    def f(X):
        tmp = X.copy()
        return model(tmp)

    # num_available_samples = len(x_test)
    # background_samples = min(background_samples, num_available_samples)
    # random_indices = np.random.choice(num_available_samples, background_samples, replace=False)
    # X_background = x_test[random_indices]
    # masker_blur = shap.maskers.Image("blur(128,128)", X_background[0].shape
    # explainer_blur = shap.Explainer(f, masker_blur, output_names=class_labels)

    blurred_image = cv2.GaussianBlur(image, (51, 51), 0)
    background = np.expand_dims(blurred_image, axis=0)
    masker = shap.maskers.Image("blur(128,128)", image.shape)
    explainer = shap.Explainer(f, masker, output_names=class_labels)

    image_batch = np.expand_dims(image, axis=0)
    preds = model.predict(image_batch, verbose=0)
    predicted_class_idx = np.argmax(preds[0])

    if specific_classes is not None:
        output_indices = specific_classes
    elif top_k is not None:
        output_indices = shap.Explanation.argsort.flip[:top_k]
    else:
       # output_indices = [predicted_class_idx]
        output_indices = shap.Explanation.argsort.flip[:len(class_labels)]

    # shap_values = explainer_blur(image_batch, max_evals=500, batch_size=50, outputs=output_indices)
    shap_values = explainer(image_batch, max_evals=200, batch_size=50, outputs=output_indices) # 500

    #shap.image_plot(shap_values)
    return shap_values, predicted_class_idx, preds


def get_shap_heatmap(shap_values):
    shap_array = shap_values.values[0, :, :, 0]
    shap_2d = shap_array.sum(axis=-1)

    shap_2d_norm = (shap_2d - shap_2d.min()) / (shap_2d.max() - shap_2d.min() + 1e-8)

    return shap_2d_norm


def get_shap_overlay(original_image, shap_val, alpha=0.4):
    shap_values = shap_val[0].values  
    if shap_values.ndim == 4:
        shap_values = shap_values.squeeze() 

    if shap_values.ndim == 3 and shap_values.shape[-1] == 3:
        map_2d = shap_values.sum(axis=-1)  # (H, W)
    else:
        map_2d = shap_values

    pos = np.clip(map_2d, 0, None)
    neg = -np.clip(map_2d, None, 0)

    if pos.max() > 0:
        pos /= pos.max()
    if neg.max() > 0:
        neg /= neg.max()


    colored_shap = np.stack([pos, np.zeros_like(pos), neg], axis=-1)  # (H, W, 3)

    overlay = (1 - alpha) * original_image + alpha * colored_shap
    overlay = np.clip(overlay, 0, 1)

    return overlay
 