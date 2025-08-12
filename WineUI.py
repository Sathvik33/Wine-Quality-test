import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(dir_path, "Model", "svc_model.pkl")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# UI Title
st.title("🍷 Wine Quality Prediction App")
st.markdown("""This app predicts the **quality** of wine based on its physicochemical properties.\
Fill in the values below and click **Predict**!
""")

def user_input_features():
    fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=7.0)
    volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.7)
    citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.0)
    residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, value=1.9)
    chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, value=0.076)
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=11.0)
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=34.0)
    density = st.number_input('Density', min_value=0.9900, max_value=1.0050, value=0.9978, format="%.4f")
    pH = st.number_input('pH', min_value=2.0, max_value=4.5, value=3.51)
    sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.56)
    alcohol = st.number_input('Alcohol', min_value=8.0, max_value=15.0, value=9.4)
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)
    st.success(f"Predicted Wine Quality: {prediction[0]}")