import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from core.services import CropRecommendationService
import plotly.express as px

st.set_page_config(page_title="🌱 Smart Crop Recommendation", page_icon="🌾", layout="wide")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("### Get the best crop suggestion based on your farm conditions.")

service = CropRecommendationService()

col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 40)
    K = st.number_input("Potassium (K)", 0, 200, 40)
with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
with col3:
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("🌱 Recommend Crop", use_container_width=True):
    crop = service.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f"✅ Recommended Crop: **{crop.capitalize()}**")
    st.plotly_chart(px.bar(
        x=["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        y=[N, P, K, temperature, humidity, ph, rainfall],
        title="Your Farm Condition Overview"
    ), use_container_width=True)
