# ============================================
# Heart Disease Prediction - train.py
# ============================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


# ============================================
# End
# ============================================

print("\nHeart Disease Prediction Pipeline Completed Successfully!")