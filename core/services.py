from core.models import load_model


class CropRecommendationService:
    def __init__(self):
        self.model = load_model()

    def recommend_crop(self, N: float, P: float, K: float, temperature: float,
                       humidity: float, ph: float, rainfall: float) -> str:
        prediction = self.model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        return prediction[0]
