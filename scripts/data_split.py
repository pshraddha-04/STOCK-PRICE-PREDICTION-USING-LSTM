import pandas as pd
import os

data_path = os.path.join("data", "microsoft_data_SMA_RSI_BBands_MACD.csv")
train_path = os.path.join("data", "train.csv")
test_path = os.path.join("data", "test.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42  # for reproducibility

# Load dataset with error handling
try:
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    print(f"[SUCCESS] Loaded dataset: {df.shape}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Chronological split
    split_index = int(len(df) * (1 - TEST_SIZE))
    
    if split_index <= 0 or split_index >= len(df):
        raise ValueError(f"Invalid split index: {split_index}")
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print(f"[SUCCESS] Train set shape: {train_df.shape}")
    print(f"[SUCCESS] Test set shape: {test_df.shape}")
    
    # Save with error handling
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=True)
    test_df.to_csv(test_path, index=True)
    
    print(f"[SUCCESS] Data split completed successfully")
    
except Exception as e:
    print(f"[ERROR] Error in data splitting: {e}")
    exit(1)