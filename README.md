## 📈 Stock Data Preparation for LSTM (Microsoft)

This repository prepares Microsoft (MSFT) historical price data and computes common technical indicators for downstream modeling (e.g., LSTM). The main script fetches data with yfinance, cleans it, and generates features including SMA, RSI, Bollinger Bands, and MACD.

---

## 📂 Project Structure
```bash
STOCK-PRICE-PREDICTION-USING-LSTM/
├── data/
│   ├── microsoft_stock_raw.csv                     # Raw download
│   ├── microsoft_stock_clean.csv                   # Cleaned data
│   └── microsoft_data_SMA_RSI_BBands_MACD.csv      # Final feature set
│
├── scripts/
│   └── msft_data_prep.py                           # Data prep pipeline
│
├── requirement.txt
└── README.md
```

---

## ⚙️ Requirements
- Python 3.9+

Install dependencies:
```bash
pip install -r requirement.txt
```

Optional virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

---

## 🚀 How to Run
Run the Microsoft data prep script:
```bash
python scripts/msft_data_prep.py
```

Outputs (saved under `data/`):
- `microsoft_stock_raw.csv`
- `microsoft_stock_clean.csv`
- `microsoft_data_SMA_RSI_BBands_MACD.csv`

What the script does:
- Downloads MSFT OHLCV with `auto_adjust=False`
- Cleans data: interpolate missing points, remove duplicates, align to business days
- Adds indicators: SMA_20, SMA_50, RSI_14, Bollinger Bands, MACD (12-26-9)
- Plots Close with SMA overlays

---

## 🔧 Customize
- Change ticker and dates in `scripts/msft_data_prep.py`:
  - `ticker_symbol = "MSFT"`
  - `start_date = "YYYY-MM-DD"`
  - `end_date = "YYYY-MM-DD"`
- You can add/remove indicators by editing the bottom sections of the script (SMA/RSI/BBands/MACD).

---

## 🔄 Update/Refresh Data
- Re-run the script with a newer `end_date` (or update both `start_date` and `end_date`).
- The pipeline will overwrite intermediate files and regenerate the final `microsoft_data_SMA_RSI_BBands_MACD.csv`.

---

## 🛠️ Troubleshooting
- yfinance timeouts: re-run the script; ensure a stable connection.
- File in use on Windows: close any open CSV in Excel/Viewers before re-running.

---

## 📌 Notes
- The outputs are suitable as inputs for LSTM or other forecasting models.
- Feel free to adapt the script for other tickers and indicators.
