from core.train_crop_recommendation_model import train_and_save_model, find_dataset
from pathlib import Path

if __name__ == "__main__":
    dataset_path = find_dataset("dataset/crop_recommendation/crop_recommendation.csv")
    model_dir = Path("models")
    train_and_save_model(dataset_path, model_dir=model_dir)
    print("Model training completed. Run `streamlit run ui/app.py` to launch the UI.")
