import pandas as pd
import os
import matplotlib.pyplot as plt

# Load cleaned dataset
folder_path = "data"
clean_file_path = os.path.join(folder_path, "microsoft_stock_clean.csv")
data = pd.read_csv(clean_file_path, parse_dates=["Date"], index_col="Date")

# SMA
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

# RSI
window_length = 14
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
avg_loss = loss.rolling(window=window_length, min_periods=1).mean()

rs = avg_gain / avg_loss
data["RSI_14"] = 100 - (100 / (1 + rs))

# Bollinger Bands
bb_window = 20
data["BB_Middle"] = data["Close"].rolling(window=bb_window).mean()
data["BB_Std"] = data["Close"].rolling(window=bb_window).std()
data["BB_Upper"] = data["BB_Middle"] + (2 * data["BB_Std"])
data["BB_Lower"] = data["BB_Middle"] - (2 * data["BB_Std"])
data.drop(columns=["BB_Std"], inplace=True)

# MACD
ema_short = data["Close"].ewm(span=12, adjust=False).mean()
ema_long = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema_short - ema_long
data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

# Saving dataset with indicators
final_file_path = os.path.join(folder_path, "microsoft_data_SMA_RSI_BBands_MACD.csv")
data.to_csv(final_file_path, index=True)
print(f"\nFinal dataset with SMA, RSI, Bollinger Bands, and MACD saved to: {final_file_path}")


