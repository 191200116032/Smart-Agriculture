# ui/app.py
import sys
from pathlib import Path

# ----------------- Ensure project root is in sys.path -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ----------------- Imports -----------------
import streamlit as st
import plotly.express as px
import pandas as pd

from core.services import CropRecommendationService
from core.yield_services import CropYieldPredictionService
from core.disease_detection_service import CropDiseaseDetectionService

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="🌾 Smart Agriculture Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Agriculture Dashboard")
st.markdown("Empowering farmers with AI-driven insights for better decision making.")

# ----------------- Tabs -----------------
tabs = st.tabs([
    "🌾 Crop Recommendation",
    "📈 Crop Yield Prediction",
    "🩺 Disease Detection"
])

# ----------------- Crop Recommendation -----------------
with tabs[0]:
    st.header("🌾 Crop Recommendation System")
    st.write("Get the best crop suggestion based on your farm conditions.")

    rec_service = CropRecommendationService()

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
        crop = rec_service.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"✅ Recommended Crop: **{crop.capitalize()}**")

        fig = px.bar(
            x=["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
            y=[N, P, K, temperature, humidity, ph, rainfall],
            title="Farm Condition Overview",
            labels={"x": "Features", "y": "Values"},
            text=[N, P, K, temperature, humidity, ph, rainfall]
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Crop Yield Prediction -----------------
with tabs[1]:
    st.header("📈 Crop Yield Prediction")
    st.write("Estimate crop yield based on location, crop type, and production data.")

    yield_service = CropYieldPredictionService()

    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.selectbox("Select State", ["Karnataka", "Maharashtra", "Punjab", "Tamil Nadu"])
        crop = st.selectbox("Select Crop", ["Rice", "Wheat", "Maize", "Cotton"])
    with col2:
        year = st.number_input("Year", min_value=2000, max_value=2025, value=2015)
        area = st.number_input("Area (hectares)", min_value=1, value=1000)
    with col3:
        production = st.number_input("Production (tons)", min_value=1, value=3000)

    if st.button("📊 Predict Yield", use_container_width=True):
        # Actual yield
        actual_yield_hg_ha = (production * 10000) / area
        actual_yield_tons_ha = actual_yield_hg_ha / 10000

        # Predicted yield
        predicted_yield_hg_ha = yield_service.predict_yield(
            state, crop, year, area, production,
            rainfall=1500, pesticides=120, temperature=25
        )
        predicted_yield_tons_ha = predicted_yield_hg_ha / 10000

        st.success(f"🌱 **Predicted Yield:** {predicted_yield_tons_ha:.2f} tons/hectare")
        st.info(f"📊 **Actual Yield (calculated from input):** {actual_yield_tons_ha:.2f} tons/hectare")

        df_yield = pd.DataFrame({
            "Type": ["Actual", "Predicted"],
            "Yield (tons/ha)": [actual_yield_tons_ha, predicted_yield_tons_ha]
        })
        fig2 = px.bar(
            df_yield,
            x="Type", y="Yield (tons/ha)",
            text="Yield (tons/ha)",
            title=f"Yield Comparison for {crop} ({state}, {year})"
        )
        fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

# ----------------- Disease Detection -----------------
with tabs[2]:
    st.header("🩺 Crop Disease Detection")
    st.write("Upload a leaf image to detect crop diseases automatically.")

    # Load cached model
    disease_service = CropDiseaseDetectionService()

    # Upload image
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        result = disease_service.predict_disease(uploaded_file)

        # Replace class indices with meaningful class names
        class_names = [
            "Cassava Bacterial Blight",
            "Cassava Brown Streak Disease",
            "Cassava Green Mite Damage",
            "Cassava Mosaic Disease",
            "Healthy"
        ]
        predicted_class_name = class_names[int(result['predicted_class'].split()[-1])]

        st.success(f"✅ Predicted Disease: **{predicted_class_name}**")
