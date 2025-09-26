from core.crop_recommendation_utils import load_model
import pandas as pd


class CropRecommendationService:
    def __init__(self):
        self.model = load_model()

    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        # Create a DataFrame with the same feature names as training
        X_input = pd.DataFrame([{
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }])

        # Add interaction features used in training
        X_input['temp_humidity'] = X_input['temperature'] * X_input['humidity']
        X_input['rainfall_per_N'] = X_input['rainfall'] / (X_input['N'] + 1e-5)

        # Predict using the model
        return self.model.predict(X_input)[0]
