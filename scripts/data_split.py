import pandas as pd
import os

data_path = os.path.join("data", "microsoft_data_SMA_RSI_BBands_MACD.csv")
train_path = os.path.join("data", "train.csv")
test_path = os.path.join("data", "test.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42  # for reproducibility

# Load dataset
df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
print(f"Loaded dataset: {df.shape}")

# Chronological split
split_index = int(len(df) * (1 - TEST_SIZE))
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

train_df.to_csv(train_path, index=True)
test_df.to_csv(test_path, index=True)