from pathlib import Path
import joblib

# Go one level up from 'core' to reach the project root
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crop_recommendation_saved_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please run train_crop_recommendation_model.py first."
        )
    model_data = joblib.load(MODEL_PATH)
    if isinstance(model_data, dict):
        return model_data['model']
    return model_data
