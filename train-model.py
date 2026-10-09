import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Real Kaggle Dataset
try:
    df = pd.read_csv("heart.csv")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please download it from Kaggle and place it in this folder.")
    exit()

print("Dataset shape:", df.shape)
print("Columns available:", df.columns.tolist())

# Split features and target ('target' or 'num' depending on exact CSV version, usually named 'target')
target_col = 'target' if 'target' in df.columns else df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

# 2. Scale features for neural network optimization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Define Keras Sequential ANN Model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training Keras ANN on Kaggle data...")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 4. Save artifacts
model.save("heart_ann_model.h5")
joblib.dump(scaler, "heart_scaler.pkl")
print("Model saved as 'heart_ann_model.h5' and scaler as 'heart_scaler.pkl'.")