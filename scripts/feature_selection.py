# scripts/feature_selection.py

import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "microsoft_data_SMA_RSI_BBands_MACD.csv")

# Load dataset
data = pd.read_csv(INPUT_FILE, index_col="Date")
print("Original dataset shape:", data.shape)

# correlation matrix
corr_matrix = data.corr().abs()  # absolute correlation
print("\nCorrelation matrix:\n", corr_matrix)

threshold = 0.95

# Upper triangle to avoid duplicates
upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop highly correlated features
data_selected = data.drop(columns=to_drop)

print("Selected features :", data_selected.columns.tolist())


