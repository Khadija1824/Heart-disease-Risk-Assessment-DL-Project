import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Heart Disease Prediction App", page_icon="🫀", layout="centered")

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("heart_ann_model.h5")
    scaler = joblib.load("heart_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please run `train_model.py` first.")
    st.stop()

st.title("🫀 AI-Powered Heart Disease Risk Assessment")
st.markdown("Predict cardiac health risk using a deep learning model trained on real clinical Kaggle data.")

# Sidebar Inputs matching standard UCI Heart Disease features
st.sidebar.header("Patient Clinical Metrics")
age = st.sidebar.slider("Age", 29, 77, 54)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible)", options=[1, 2, 3])

if st.button("Evaluate Heart Risk", type="primary"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    
    prediction_prob = model.predict(input_scaled)[0][0]
    
    st.subheader("Diagnostic Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Disease Probability", value=f"{prediction_prob * 100:.2f}%")
        
    with col2:
        if prediction_prob > 0.5:
            st.error("High Risk: Model indicates indicators associated with heart disease. Consult a cardiologist.")
        else:
            st.success("Low Risk: Clinical indicators appear within safer parameters.")