
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION & ENTERPRISE CAR STORE CSS ---
st.set_page_config(page_title="AutoMax | Enterprise MPG Predictor", layout="wide", page_icon="🏎️")

st.markdown("""
    <style>
    /* Premium High-Res Dark Automotive Showroom Background */
    .stApp {
        background: linear-gradient(rgba(5, 7, 12, 0.82), rgba(5, 7, 12, 0.90)), 
                    url('https://images.unsplash.com/photo-1617814076367-b759c7d7e738?auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Modern Glassmorphism Cards */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(16, 22, 34, 0.70) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }

    /* Vibrant Typography Styling */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    p, span, label {
        color: #e2e8f0 !important;
    }

    /* Metric Widget Styling */
    div[data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }

    /* Custom Sidebar Dark Theme */
    section[data-testid="stSidebar"] {
        background-color: rgba(7, 10, 18, 0.92) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    linear_model = joblib.load('linear_model.pkl')
    ridge_grid = joblib.load('ridge_model.pkl')
    lasso_grid = joblib.load('lasso_model.pkl')
    return scaler, linear_model, ridge_grid, lasso_grid

try:
    scaler,linear_model, ridge_grid,lasso_grid = load_assets()

except Exception as e:
    st.error("⚠️ Model or Scaler files not found! Make sure scaler.pkl, linear_model.pkl, ridge_model.pkl, and lasso_model.pkl are in the project folder.")
    st.stop()

st.title(" AutoMax MPG Multi-Model Predictor")
st.write("Enter vehicle specifications in the sidebar to compare predictions across Linear, Ridge, and Lasso Regression models.")
st.sidebar.header("🛠️ Vehicle Specifications")
cylinders = st.sidebar.number_input("Cylinders", min_value=3.0, max_value=8.0, value=4.0, step=1.0)
displacement = st.sidebar.number_input("Displacement", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
horsepower = st.sidebar.number_input("Horsepower", min_value=40.0, max_value=300.0, value=95.0, step=1.0)
weight = st.sidebar.number_input("Weight (lbs)", min_value=1500.0, max_value=6000.0, value=2800.0, step=50.0)
acceleration = st.sidebar.number_input("Acceleration", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
model_year = st.sidebar.number_input("Model Year (e.g. 76 for 1976)", min_value=70.0, max_value=82.0, value=76.0,
                                     step=1.0)

# --- 3. INFERENCE PIPELINE ---
new_car = pd.DataFrame({
    "cylinders": [cylinders],
    "displacement": [displacement],
    "horsepower": [horsepower],
    "weight": [weight],
    "acceleration": [acceleration],
    "model year": [model_year]
})

# Feature Scaling
new_car_scaled = scaler.transform(new_car)

# Predictions
predicted_mpg_linear = linear_model.predict(new_car_scaled)[0]
predicted_mpg_ridge = ridge_grid.predict(new_car_scaled)[0]
predicted_mpg_lasso = lasso_grid.predict(new_car_scaled)[0]

# Dynamic Car Category Image based on Average Prediction
avg_mpg = (predicted_mpg_linear + predicted_mpg_ridge + predicted_mpg_lasso) / 3


def get_car_category(mpg):
    if mpg >= 30:
        return "Eco Compact Series", "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=800&q=80", "#2ecc71"
    elif mpg >= 20:
        return "Executive Family Sedan", "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=800&q=80", "#f39c12"
    else:
        return "High-Performance V8 Muscle", "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=800&q=80", "#e74c3c"


car_class, car_img_url, theme_color = get_car_category(avg_mpg)

# --- 4. DASHBOARD DISPLAY ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Model Predictions Comparison")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Linear Model", f"{predicted_mpg_linear:.2f} MPG")
    m_col2.metric("Ridge Model", f"{predicted_mpg_ridge:.2f} MPG")
    m_col3.metric("Lasso Model", f"{predicted_mpg_lasso:.2f} MPG")

    st.write("---")
    st.markdown(
        f"**Matched Showroom Category:** <span style='color:{theme_color}; font-size:20px; font-weight:bold;'>{car_class}</span>",
        unsafe_allow_html=True)
    st.image(car_img_url, caption=f"Typical vehicle profile matching ~{avg_mpg:.1f} MPG rating",
             use_container_width=True)

with col2:
    st.subheader("📋 Input Summary")
    st.dataframe(new_car, use_container_width=True)

    st.info(
        f"💡 **Insight:** Model predictions vary slightly based on regularization coefficients (L1/L2 penalties) applied during Ridge and Lasso tuning.")
