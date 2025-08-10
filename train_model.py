# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load data
df = pd.read_csv("data/personal_health_data.csv")

# Select features
features = ["Heart_Rate", "Blood_Oxygen_Level", "ECG", "Skin_Temperature", "Sleep_Duration", "Stress_Level"]
for col in features:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])

# Create risk target
df["Risk"] = np.where(df["Health_Score"] < 60, 1, 0)

X = df[features].values
y = df["Risk"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train_3d.shape[2])),
    Conv1D(32, kernel_size=1, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train_3d, y_train, validation_data=(X_test_3d, y_test), epochs=30, batch_size=32, callbacks=[es])

# Evaluate
loss, acc = model.evaluate(X_test_3d, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# Save model and scaler
model.save("models/wearable_risk_model.h5")
joblib.dump(scaler, "models/wearable_scaler.pkl")
pd.DataFrame(features, columns=["feature"]).to_csv("models/features.csv", index=False)
print("Model and artifacts saved.")
