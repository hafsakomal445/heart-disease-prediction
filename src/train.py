# ============================================
# Heart Disease Prediction - train.py
# ============================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("../data/heart.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully!\n")


# ============================================
# Create Target Variable
# ============================================

# num = 0 --> No disease
# num > 0 --> Disease

df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Drop original target column
df.drop('num', axis=1, inplace=True)

print("Target variable created!\n")


# ============================================
# Handle Missing Values
# ============================================

# Numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled!\n")


# ============================================
# Encode Categorical Features
# ============================================

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical features encoded!\n")


# ============================================
# Feature Selection
# ============================================

X = df.drop('target', axis=1)
y = df['target']

print("Features and target separated!\n")


# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-Test split completed!\n")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}\n")


# ============================================
# Train Logistic Regression Model
# ============================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!\n")


# ============================================
# Make Predictions
# ============================================

y_pred = model.predict(X_test)


# ============================================
# Evaluate Model
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("========== MODEL EVALUATION ==========")
print(f"Accuracy Score: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

print(feature_importance.sort_values(by='Coefficient', ascending=False))
# ============================================
# Random Forest Model
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest model trained successfully!\n")
# Make Predictions
rf_pred = rf_model.predict(X_test)
# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
# ============================================
# Hyperparameter Tuning
# ============================================

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("========== RANDOM FOREST EVALUATION ==========")

print(f"Random Forest Accuracy: {rf_accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, rf_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, rf_pred))
# Compare Models
print("========== MODEL COMPARISON ==========")

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
# Visual Comparison (IMPORTANT)
models = ['Logistic Regression', 'Random Forest']
scores = [accuracy, rf_accuracy]

plt.figure(figsize=(6,4))
plt.bar(models, scores)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(grid_search.best_score_)
best_rf = grid_search.best_estimator_

best_rf_pred = best_rf.predict(X_test)

best_rf_accuracy = accuracy_score(y_test, best_rf_pred)

print(f"\nTuned Random Forest Accuracy: {best_rf_accuracy:.4f}")
print("\n========== FINAL MODEL COMPARISON ==========")

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Tuned Random Forest Accuracy: {best_rf_accuracy:.4f}")
models = [
    'Logistic Regression',
    'Random Forest',
    'Tuned Random Forest'
]

scores = [
    accuracy,
    rf_accuracy,
    best_rf_accuracy
]

plt.figure(figsize=(8,5))
plt.bar(models, scores)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance)
plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")

plt.show()
# ============================================
# Save Model
# ============================================

joblib.dump(best_rf, "../model/heart_disease_model.pkl")

print("Model saved successfully!")
# ============================================
# End
# ============================================

print("\nHeart Disease Prediction Pipeline Completed Successfully!")