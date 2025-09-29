import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path

MODEL_DIR = Path("models/cropnet_saved_model")


@st.cache_resource  # caches the loaded model in Streamlit
def load_disease_model():
    print(f"✅ Loading CropNet model from {MODEL_DIR}")
    model = tf.saved_model.load(str(MODEL_DIR))

    # Check available signatures
    sigs = list(model.signatures.keys())
    print("Available signatures:", sigs)

    if "serving_default" in model.signatures:
        infer = model.signatures["serving_default"]
    else:
        infer = list(model.signatures.values())[0]

    print("✅ Model ready for inference")
    return infer


def preprocess_image(img_file, target_size=(224, 224)):
    """Preprocess uploaded image for model input"""
    image = Image.open(img_file).convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0  # normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array
