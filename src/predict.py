import joblib
import pandas as pd

# Load saved model
model = joblib.load("../model/heart_disease_model.pkl")

print("Model loaded successfully!")
sample_data = {
    'id': [1],
    'age': [52],
    'sex': [1],
    'dataset': [0],
    'cp': [2],
    'trestbps': [130],
    'chol': [250],
    'fbs': [0],
    'restecg': [1],
    'thalch': [150],
    'exang': [0],
    'oldpeak': [1.2],
    'slope': [1],
    'ca': [0],
    'thal': [2]
}

sample_df = pd.DataFrame(sample_data)
prediction = model.predict(sample_df)

if prediction[0] == 1:
    print("Prediction: Patient HAS heart disease")
else:
    print("Prediction: Patient DOES NOT have heart disease")

probability = model.predict_proba(sample_df)

print("\nPrediction Probability:")
print(probability)