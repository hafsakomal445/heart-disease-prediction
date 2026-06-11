# test_model.py

import joblib

model = joblib.load("model/heart_disease_model.pkl")

print("Model loaded successfully!")
print(type(model))