import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import tensorflow as tf
import random

# Reproducibility settings
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # Force deterministic ops if possible

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math


# Paths
DATA_DIR = "data"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
TARGET_SCALER_PATH = os.path.join(DATA_DIR, "scalers", "Close_scaler.pkl")

# Load sequences with error handling
try:
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(y_TEST_PATH)
    print("[SUCCESS] Training data loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading training data: {e}")
    exit(1)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(units=25, activation="relu"),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Predictions
target_scaler = joblib.load(TARGET_SCALER_PATH)

# Test predictions
y_pred_test = model.predict(X_test)
y_test_inv = target_scaler.inverse_transform(y_test)
y_pred_test_inv = target_scaler.inverse_transform(y_pred_test)

# Train predictions
y_pred_train = model.predict(X_train)
y_train_inv = target_scaler.inverse_transform(y_train)
y_pred_train_inv = target_scaler.inverse_transform(y_pred_train)

# Training Metrics
mse_train = mean_squared_error(y_train_inv, y_pred_train_inv)
rmse_train = math.sqrt(mse_train)
mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
r2_train = r2_score(y_train_inv, y_pred_train_inv)

print("\nTraining Metrics:")
print(f"RMSE: {rmse_train:.6f}")
print(f"MAE:  {mae_train:.6f}")
print(f"R²:   {r2_train:.6f}")

# Test Metrics
mse_test = mean_squared_error(y_test_inv, y_pred_test_inv)
rmse_test = math.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_inv, y_pred_test_inv)
r2_test = r2_score(y_test_inv, y_pred_test_inv)

print("\nTest Metrics:")
print(f"RMSE: {rmse_test:.6f}")
print(f"MAE:  {mae_test:.6f}")
print(f"R²:   {r2_test:.6f}")

# Visualization
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label="Actual Price (Test)")
plt.plot(y_pred_test_inv, label="Predicted Price (Test)")
plt.title("LSTM Model: Actual vs Predicted (Test Data)")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Optional: Visualize training fit
plt.figure(figsize=(10,6))
plt.plot(y_train_inv, label="Actual Price (Train)")
plt.plot(y_pred_train_inv, label="Predicted Price (Train)")
plt.title("LSTM Model: Actual vs Predicted (Train Data)")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()