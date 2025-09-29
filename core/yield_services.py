import joblib
import pandas as pd
from pathlib import Path


class CropYieldPredictionService:
    def __init__(self):
        model_path = Path("models/crop_yield_saved_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError("Crop yield model not found. Train it first.")
        self.model, self.feature_names = joblib.load(model_path)

    def predict_yield(self, state, crop, year, area, production, rainfall, pesticides, temperature):
        # Convert production + area into yield (hg/ha)
        yield_hg_ha = (production * 10000) / area

        df = pd.DataFrame([{
            "Yield": yield_hg_ha,
            "Year": year,
            "Rainfall": rainfall,
            "Pesticides": pesticides,
            "Temperature": temperature,
            "State": state,
            "Crop": crop
        }])

        # One-hot encode categorical variables to match training columns
        df = pd.get_dummies(df, columns=["State", "Crop"], drop_first=True)

        # Add missing columns (due to unseen states/crops) to match training features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]

        prediction = self.model.predict(df)[0]
        return prediction
