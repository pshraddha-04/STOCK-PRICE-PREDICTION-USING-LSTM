import numpy as np
import os
import joblib
import tensorflow as tf
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
DATA_DIR = "data"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
TARGET_SCALER_PATH = os.path.join(DATA_DIR, "scalers", "Close_scaler.pkl")
RESULTS_PATH = os.path.join(DATA_DIR, "random_search_results.csv")

# Load Data
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(y_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)
y_test = np.load(y_TEST_PATH)

target_scaler = joblib.load(TARGET_SCALER_PATH)

# Load best hyperparameters from Random Search
results_df = pd.read_csv(RESULTS_PATH)
best_params = results_df.sort_values(by="rmse").iloc[0].to_dict()

print("Best Hyperparameters Found:", best_params)

# Build model using best parameters
def build_model(lstm_units, dropout, dense_units, learning_rate):
    model = Sequential([
        LSTM(units=int(lstm_units), return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(float(dropout)),
        Dense(int(dense_units), activation="relu"),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=float(learning_rate))
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

model = build_model(
    lstm_units=best_params["lstm_units"],
    dropout=best_params["dropout"],
    dense_units=best_params["dense_units"],
    learning_rate=best_params["learning_rate"]
)

# Train with best batch size
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=int(best_params["batch_size"]),
    validation_data=(X_test, y_test),
    verbose=1
)

# Predictions
y_pred_test = model.predict(X_test)
y_test_inv = target_scaler.inverse_transform(y_test)
y_pred_test_inv = target_scaler.inverse_transform(y_pred_test)

# Save predictions
predictions_df = pd.DataFrame({
    "Actual": y_test_inv.flatten(),
    "Predicted": y_pred_test_inv.flatten()
})
predictions_path = os.path.join(DATA_DIR, "test_predictions.csv")
predictions_df.to_csv(predictions_path, index=False)

print("\n Predictions saved to:", predictions_path)

# Evaluation Metrics
rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
r2 = r2_score(y_test_inv, y_pred_test_inv)

print("\nTest Set Evaluation with Fine-Tuned Model:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R²:   {r2:.6f}")