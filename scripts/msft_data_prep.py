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
# Step 7: Calculate Simple Moving Averages (SMA)
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

# Step 8: Save dataset with SMA columns
sma_file_path = os.path.join(folder_path, "microsoft_stock_with_sma_20_50.csv")
data.to_csv(sma_file_path, index=True)
print(f"\nMicrosoft stock data with SMA saved to: {sma_file_path}")

import matplotlib.pyplot as plt

# Step 9: Plot Close price with SMA lines
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Close"], label="Close Price", linewidth=1)
plt.plot(data.index, data["SMA_20"], label="20-Day SMA", linewidth=1.2)
plt.plot(data.index, data["SMA_50"], label="50-Day SMA", linewidth=1.2)

plt.title("Microsoft Stock Price with 20 & 50 Day SMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Calculate Relative Strength Index (RSI)
window_length = 14
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
avg_loss = loss.rolling(window=window_length, min_periods=1).mean()

rs = avg_gain / avg_loss
data["RSI_14"] = 100 - (100 / (1 + rs))

# Step 11: Calculate Bollinger Bands
bb_window = 20
data["BB_Middle"] = data["Close"].rolling(window=bb_window).mean()
data["BB_Std"] = data["Close"].rolling(window=bb_window).std()
data["BB_Upper"] = data["BB_Middle"] + (2 * data["BB_Std"])
data["BB_Lower"] = data["BB_Middle"] - (2 * data["BB_Std"])
data.drop(columns=["BB_Std"], inplace=True)  # keep dataset clean

# Step 12: Calculate MACD (12-26 EMA with 9 signal line)
ema_short = data["Close"].ewm(span=12, adjust=False).mean()
ema_long = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema_short - ema_long
data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

# Save updated dataset (overwrite old SMA file first)
data.to_csv(sma_file_path, index=True)
print(f"\nUpdated dataset with SMA, RSI, Bollinger Bands, and MACD saved to: {sma_file_path}")

# Rename the file to a more descriptive name
final_file_path = os.path.join(folder_path, "microsoft_data_SMA_RSI_BBands_MACD.csv")
os.rename(sma_file_path, final_file_path)
print(f"File renamed to: {final_file_path}")
