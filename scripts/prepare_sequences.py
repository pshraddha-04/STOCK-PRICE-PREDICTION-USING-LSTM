import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Paths
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "microsoft_data_SMA_RSI_BBands_MACD.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SCALER_DIR = os.path.join(DATA_DIR, "scalers")
os.makedirs(SCALER_DIR, exist_ok=True)

# Parameters
LOOKBACK = 60   
TEST_RATIO = 0.2  

# Load original dataset
data = pd.read_csv(INPUT_FILE, index_col="Date")

data = data.dropna()  # remove first row with NaN

# Define input features and target
input_features = ["Close", "Open", "Volume", "RSI_14", "MACD", "MACD_Hist"]

target_feature = "Close"

X_data = data[input_features].copy()
y_data = data[target_feature].copy().values.reshape(-1, 1)

# Scale input features individually
scalers = {}
X_scaled = np.zeros_like(X_data.values)
for i, col in enumerate(input_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled[:, i] = scaler.fit_transform(X_data[[col]]).flatten()
    scalers[col] = scaler
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{col}_scaler.pkl"))

# Scale target separately
target_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = target_scaler.fit_transform(y_data)
joblib.dump(target_scaler, os.path.join(SCALER_DIR, f"{target_feature}_scaler.pkl"))

# Train-test split
split_idx = int(len(X_scaled) * (1 - TEST_RATIO))
X_train_raw, X_test_raw = X_scaled[:split_idx], X_scaled[split_idx:]
y_train_raw, y_test_raw = y_scaled[:split_idx], y_scaled[split_idx:]

# LSTM sequences
def create_sequences(data, lookback, target_data=None):
    X_seq, y_seq = [], []
    for i in range(lookback, len(data)):
        X_seq.append(data[i - lookback:i])
        if target_data is not None:
            y_seq.append(target_data[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq) if target_data is not None else None
    return X_seq, y_seq

X_train, y_train = create_sequences(X_train_raw, LOOKBACK, y_train_raw)
X_test, y_test = create_sequences(X_test_raw, LOOKBACK, y_test_raw)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Save sequences 
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

train_df = data.iloc[:split_idx].copy()
test_df = data.iloc[split_idx:].copy()
train_df.to_csv(TRAIN_PATH, index=True)
test_df.to_csv(TEST_PATH, index=True)

print("Data preparation complete. Sequences saved for LSTM training.")