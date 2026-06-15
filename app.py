import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")

model = joblib.load("model/heart_disease_model.pkl")

# Input fields
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
trestbps = st.number_input("Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)

if st.button("Predict"):
    st.success("Prediction button clicked!")