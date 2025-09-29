from core.crop_disease_utils import load_disease_model, preprocess_image
import tensorflow as tf
import numpy as np


class CropDiseaseDetectionService:
    def __init__(self):
        # load model once (cached by Streamlit)
        self.infer = load_disease_model()

    def predict_disease(self, img_file):
        img_array = preprocess_image(img_file)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        outputs = self.infer(img_tensor)

        # TF SavedModel output is a dict of tensors
        preds = list(outputs.values())[0].numpy()  # shape: (1, num_classes)
        class_idx = int(np.argmax(preds))
        confidence = float(preds[0][class_idx])

        return {
            "predicted_class": f"Class {class_idx}",  # replace with real class names if available
            "confidence": confidence
        }
