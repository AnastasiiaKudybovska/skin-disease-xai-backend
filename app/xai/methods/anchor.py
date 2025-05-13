import numpy as np
from alibi.explainers import AnchorImage

def generate_anchor_for_image(image, model, threshold=0.95, p_sample=.5, n_segments=11): # 15
    predict_fn = lambda x: model.predict(x)
    
    image_batch = np.expand_dims(image, axis=0)
    preds = predict_fn(image_batch)
    predicted_class_idx = np.argmax(preds[0])

    explainer = AnchorImage(predict_fn, (224, 224, 3), segmentation_fn='slic', 
                        segmentation_kwargs={'n_segments': n_segments, 'compactness': 20, 'sigma': .5}, 
                        images_background=None)
    explanation = explainer.explain(image, threshold=.95, p_sample=p_sample, tau=0.25)

    return explanation, predicted_class_idx, preds