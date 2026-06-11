import streamlit as st
import joblib

st.title("❤️ Heart Disease Prediction")

try:
    model = joblib.load("model/heart_disease_model.pkl")

    st.success("Model loaded successfully!")

    st.write(type(model))

except Exception as e:
    st.error(f"Error loading model: {e}")