import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("data/heart.csv")

print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")

# =====================================
# Clean Columns
# =====================================

df.columns = df.columns.str.strip()

# =====================================
# Create Binary Target
# =====================================

df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df.drop("num", axis=1, inplace=True)

print("Target variable created.")

# =====================================
# Handle Missing Values
# =====================================

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled.")

# =====================================
# Encode Categorical Features
# =====================================

encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

print("Categorical features encoded.")

# =====================================
# Features and Target
# =====================================

X = df.drop("target", axis=1)
y = df["target"]

# =====================================
# Train/Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train/Test split completed.")

# =====================================
# Train Model
# =====================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")

# =====================================
# Predictions
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================
# Feature Importance
# =====================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features:")
print(importance_df.head(10))

# =====================================
# Save Model
# =====================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/heart_disease_model.pkl"
)

joblib.dump(
    list(X.columns),
    "model/feature_columns.pkl"
)

print("\nModel saved successfully.")
print("Saved: model/heart_disease_model.pkl")

print("\nPipeline completed successfully!")