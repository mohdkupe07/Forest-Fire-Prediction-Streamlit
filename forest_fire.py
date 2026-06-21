import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Algerian_forest_fires_dataset.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Convert numeric columns
non_class_cols = [col for col in df.columns if col.lower() != "classes"]

for col in non_class_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
        errors='coerce'
    )

# Remove missing values
df = df.dropna(subset=non_class_cols)

# Remove unnecessary columns
df.drop(columns=["day", "month", "year"], inplace=True)

# Encode target column
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["Classes"] = encoder.fit_transform(df["Classes"])

# Split data
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Classes"])
Y = df["Classes"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)


# Prediction
Y_pred = model.predict(X_test)


# Accuracy
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(Y_test, Y_pred)

print("Model Accuracy:", accuracy)

print(
    classification_report(Y_test, Y_pred)
)


# Save model
import joblib

joblib.dump(model, "forest_fire_model.pkl")

print("Model saved successfully!")