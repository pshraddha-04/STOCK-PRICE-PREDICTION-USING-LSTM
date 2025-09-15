import pandas as pd
import os
import matplotlib.pyplot as plt

# final dataset with indicators
folder_path = "data"
final_file_path = os.path.join(folder_path, "microsoft_data_SMA_RSI_BBands_MACD.csv")
data = pd.read_csv(final_file_path, parse_dates=["Date"], index_col="Date")

# Close Price + SMA
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

# RSI
plt.figure(figsize=(12, 4))
plt.plot(data.index, data["RSI_14"], label="RSI (14)", color="purple", linewidth=1)
plt.axhline(70, linestyle="--", color="red", alpha=0.7)
plt.axhline(30, linestyle="--", color="green", alpha=0.7)
plt.title("Relative Strength Index (RSI 14)")
plt.xlabel("Date")
plt.ylabel("RSI Value")
plt.legend()
plt.grid(True)
plt.show()

# Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Close"], label="Close Price", linewidth=1)
plt.plot(data.index, data["BB_Middle"], label="Middle Band (20 SMA)", linewidth=1.2)
plt.plot(data.index, data["BB_Upper"], label="Upper Band", linestyle="--", linewidth=1)
plt.plot(data.index, data["BB_Lower"], label="Lower Band", linestyle="--", linewidth=1)
plt.fill_between(data.index, data["BB_Lower"], data["BB_Upper"], color="gray", alpha=0.2)
plt.title("Bollinger Bands (20 Day)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# MACD
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["MACD"], label="MACD", color="blue", linewidth=1)
plt.plot(data.index, data["MACD_Signal"], label="Signal Line", color="orange", linewidth=1)
plt.bar(data.index, data["MACD_Hist"], label="MACD Histogram", color="gray", alpha=0.5)
plt.title("MACD (12, 26, 9)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
