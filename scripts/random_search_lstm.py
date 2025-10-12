import numpy as np
import os
import joblib
import tensorflow as tf
import random

# Reproducibility settings
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths
DATA_DIR = "data"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
TARGET_SCALER_PATH = os.path.join(DATA_DIR, "scalers", "Close_scaler.pkl")

BEST_MODEL_DIR = "models"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_lstm_model.keras")  # path to save best model

# Load data with error handling
try:
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(y_TEST_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    print("[SUCCESS] Data loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    exit(1)

# Hyperparameter search space
param_distributions = {
    "lstm_units": [50, 80, 100, 120, 150],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "dense_units": [25, 50, 75, 100],
    "learning_rate": [0.0005, 0.001, 0.002, 0.005],
    "batch_size": [16, 32, 64]
}

N_TRIALS = 20  

# Function to build model
def build_model(lstm_units, dropout, dense_units, learning_rate):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout),
        Dense(units=dense_units, activation="relu"),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

# Random Search Loop
results = []
best_rmse = float("inf")  # track best RMSE
best_params_saved = None   # store best hyperparameters

for trial in range(N_TRIALS):
    # Randomly pick parameters
    params = {key: random.choice(values) for key, values in param_distributions.items()}
    print(f"\n Trial {trial+1}/{N_TRIALS} with params: {params}")

    # Build and train model
    model = build_model(
        lstm_units=params["lstm_units"],
        dropout=params["dropout"],
        dense_units=params["dense_units"],
        learning_rate=params["learning_rate"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=params["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_test_inv = target_scaler.inverse_transform(y_pred_test)

    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
    r2 = r2_score(y_test_inv, y_pred_test_inv)

    print(f" Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    results.append({
        "trial": trial+1,
        **params,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })
    
    # Save model if this trial is best so far
    if rmse < best_rmse:
        best_rmse = rmse
        best_params_saved = params.copy() 
        model.save(BEST_MODEL_PATH)
        print(f" New best model saved at {BEST_MODEL_PATH} with RMSE={best_rmse:.4f}")


# Save results with error handling
try:
    results_df = pd.DataFrame(results)
    results_path = os.path.join(DATA_DIR, "random_search_results.csv")
    results_df.to_csv(results_path, index=False)
    print("\n[SUCCESS] Random Search complete. Results saved to:", results_path)
    
    # Save best hyperparameters for reference
    if best_params_saved:
        best_params_df = pd.DataFrame([best_params_saved])
        best_params_path = os.path.join(DATA_DIR, "best_hyperparameters.csv")
        best_params_df.to_csv(best_params_path, index=False)
        print(f"[SUCCESS] Best hyperparameters saved to: {best_params_path}")
except Exception as e:
    print(f"[ERROR] Error saving results: {e}")

print(results_df.sort_values(by="rmse").head(5))  # show best 5 trials