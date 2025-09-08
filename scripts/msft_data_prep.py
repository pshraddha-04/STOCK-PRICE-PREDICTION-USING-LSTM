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

# Step 5: Save the CSV inside the folder
file_path = os.path.join(folder_path, "microsoft_stock_raw.csv")
data.to_csv(file_path, index=True)

print(f"Microsoft stock data saved to: {file_path}")
print("\nFirst 5 rows:")
print(data.head())
