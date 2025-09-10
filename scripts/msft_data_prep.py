# Import required libraries
import yfinance as yf
import pandas as pd
import os  

# Step 1: Define ticker and date range
ticker_symbol = "MSFT"
start_date = "2020-09-01"  # 5 years ago
end_date = "2025-08-31"    # today

# Step 2: Download historical data
msft = yf.Ticker(ticker_symbol)

# Keep original columns (Adj Close remains separate)
data = msft.history(start=start_date, end=end_date, auto_adjust=False)

# Step 3: Keep relevant columns
data = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# Step 4: Define folder path and create if not exists
folder_path = "data"
os.makedirs(folder_path, exist_ok=True)

# Step 5: Save the raw CSV
file_path = os.path.join(folder_path, "microsoft_stock_raw.csv")
data.to_csv(file_path, index=True)
print(f"Microsoft stock data saved to: {file_path}")

# Step 6: Data Cleaning & Completeness Check
print("\nMissing values per column before cleaning:")
print(data.isnull().sum())

# Fill missing values by time-based interpolation (if any)
data.interpolate(method='time', inplace=True)

# Remove duplicates
duplicate_count = data.index.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")
data = data[~data.index.duplicated(keep='first')]

# Ensure date index is continuous (business days)
data.index = pd.to_datetime(data.index)
all_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
missing_days = all_days.difference(data.index)
print(f"\nNumber of missing trading days: {len(missing_days)}")
if len(missing_days) > 0:
    data = data.reindex(all_days)
    data.fillna(method='ffill', inplace=True)
    print("Missing trading days filled using forward-fill.")

# Ensure correct data types
for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    data[col] = data[col].astype(float)

# Quick statistical check
print("\nData statistics after cleaning:")
print(data.describe())

# ✅ IMPORTANT FIX: name the index before saving
data.index.name = "Date"

# Save cleaned data to CSV
clean_file_path = os.path.join(folder_path, "microsoft_stock_clean.csv")
data.to_csv(clean_file_path, index=True)
print(f"\nCleaned Microsoft stock data saved to: {clean_file_path}")
