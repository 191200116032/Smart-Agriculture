from pathlib import Path
from core.train_crop_recommendation_model import train_and_save_model, find_dataset
from core.train_crop_yield_model import train_and_save_yield_model


def main():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # ------------------ Crop Recommendation ------------------
    crop_rec_model_path = model_dir / "crop_recommendation_saved_model.pkl"
    if crop_rec_model_path.exists():
        print(f"✅ Crop Recommendation model already exists at {crop_rec_model_path}, skipping training.")
    else:
        dataset_path = find_dataset("dataset/crop_recommendation/crop_recommendation.csv")
        print("🚀 Training Crop Recommendation model...")
        train_and_save_model(dataset_path, model_dir=model_dir)
        print("✅ Crop Recommendation model trained and saved.")

    # ------------------ Crop Yield Prediction ------------------
    crop_yield_model_path = model_dir / "crop_yield_saved_model.pkl"
    if crop_yield_model_path.exists():
        print(f"✅ Crop Yield Prediction model already exists at {crop_yield_model_path}, skipping training.")
    else:
        print("🚀 Training Crop Yield Prediction model...")
        train_and_save_yield_model()
        print("✅ Crop Yield Prediction model trained and saved.")

    print("\n🎉 Model training check completed. Run `streamlit run ui/app.py` to launch the UI.")


if __name__ == "__main__":
    main()
