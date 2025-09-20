import numpy as np
import os
import joblib
import tensorflow as tf
import random
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reproducibility settings
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"  

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_DIR = "data"
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
TARGET_SCALER_PATH = os.path.join(DATA_DIR, "scalers", "Close_scaler.pkl")
BEST_MODEL_PATH = os.path.join("models", "best_lstm_model.keras") 
MODEL_PKL_PATH = os.path.join("models", "best_lstm_model.pkl")    
PREDICTIONS_PATH = os.path.join(DATA_DIR, "test_predictions.csv")

# Load Data
X_test = np.load(X_TEST_PATH)
y_test = np.load(y_TEST_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

# Load best model (saved from random search)
print(f"Loading best model from {BEST_MODEL_PATH} ...")
model = tf.keras.models.load_model(BEST_MODEL_PATH)  

# Save model as pkl for future use
joblib.dump(model, MODEL_PKL_PATH)  
print(f"✅ Best LSTM model saved as .pkl at: {MODEL_PKL_PATH}")

# Predictions using loaded model
y_pred_test = model.predict(X_test)

# Inverse transform to original scale
y_test_inv = target_scaler.inverse_transform(y_test)
y_pred_test_inv = target_scaler.inverse_transform(y_pred_test)

# Save predictions
predictions_df = pd.DataFrame({
    "Actual": y_test_inv.flatten(),
    "Predicted": y_pred_test_inv.flatten()
})
predictions_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"\n✅ Predictions saved to: {PREDICTIONS_PATH}")

# Evaluation Metrics
rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
r2 = r2_score(y_test_inv, y_pred_test_inv)

print("\nTest Set Evaluation using best LSTM Model:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"R²:   {r2:.6f}")