import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/heart_disease_model.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient information below.")
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

cp = st.number_input("Chest Pain Type", min_value=0, max_value=4, value=1)

trestbps = st.number_input("Resting Blood Pressure", value=120)

chol = st.number_input("Cholesterol", value=200)

thalch = st.number_input("Maximum Heart Rate", value=150)

oldpeak = st.number_input("Old Peak", value=1.0)
sex_value = 1 if sex == "Male" else 0
if st.button("Predict"):
        input_df = pd.DataFrame({
        'id': [1],
        'age': [age],
        'sex': [sex_value],
        'dataset': [0],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [0],
        'restecg': [0],
        'thalch': [thalch],
        'exang': [0],
        'oldpeak': [oldpeak],
        'slope': [1],
        'ca': [0],
        'thal': [2]
    })

prediction = model.predict(input_df)
if prediction[0] == 1:
       st.error("Patient is likely to have heart disease.")
else:
        st.success("Patient is unlikely to have heart disease.")