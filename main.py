from core.models import train_and_save_model

if __name__ == "__main__":
    train_and_save_model()
    print("Model training completed. Run `streamlit run ui/app.py` to launch the UI.")