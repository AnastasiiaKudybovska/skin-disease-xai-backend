
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2

class myGradCAM: 
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName or self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            convOutputs, predictions = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
        convOutputs, guidedGrads = convOutputs[0], guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        h, w = image.shape[1:3]
        heatmap = cv2.resize(cam.numpy(), (w, h))
        heatmap = np.maximum(heatmap, 0)
    
        numer = heatmap - np.min(heatmap)
        denom = heatmap.max() - np.min(heatmap) + eps
        heatmap = numer / denom
        
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap
        
    def overlay_heatmap(self, heatmap, image, alpha=0.5,colormap=cv2.COLORMAP_VIRIDIS): # 0.8 cv2.COLORMAP_JET cv2.COLORMAP_VIRIDIS
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap.astype(np.uint8), 1 - alpha, 0)
        return (heatmap, output)
        
    def apply_black_mask(self, heatmap, image, threshold=70, max_threshold=100):
        normalized_heatmap = heatmap / 255.0
    
        # Маска: всі значення нижче threshold будуть чорними (0)
        mask = np.zeros_like(normalized_heatmap)
    
        # Для значень від threshold до 100 робимо плавний перехід від 0 до 0.5
        transition_region = (normalized_heatmap >= threshold / 255.0) & (normalized_heatmap <= max_threshold / 255.0)
        mask[transition_region] = (normalized_heatmap[transition_region] - threshold / 255.0) / ((max_threshold - threshold) / 255.0) * 0.8
    
        # Для значень вище 100 маска = 1 (повністю видно оригінальне зображення)
        mask[normalized_heatmap > max_threshold / 255.0] = 1
    
        # Розширюємо маску до 3 каналів (RGB)
        mask = np.stack([mask] * 3, axis=-1)

        output = np.uint8(image * mask)    
        return output
    

def generate_gradcam_for_image(image, model, layer_name='block7a_project_conv', threshold=70, max_threshold=100): # conv5_block3_3_conv
    image_batch = np.expand_dims(image, axis=0)

    preds = model.predict(image_batch, verbose=0)
    predicted_class_idx = np.argmax(preds[0])

    icam = myGradCAM(model, predicted_class_idx, layer_name)
    heatmap = icam.compute_heatmap(image_batch)
    heatmap = cv2.resize(heatmap, (224, 224))

    output_with_mask = icam.apply_black_mask(heatmap, image, threshold, max_threshold)
    heatmap, output = icam.overlay_heatmap(heatmap, image, alpha=0.6)
    return predicted_class_idx, heatmap, output, output_with_mask, preds