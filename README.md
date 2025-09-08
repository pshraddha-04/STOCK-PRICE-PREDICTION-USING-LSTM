## Stock Data Prep with SMA (for LSTM)

A minimal project that fetches Reliance stock data via yfinance, computes simple moving averages (SMA20/50/200), and saves a ready-to-model CSV. You can use the generated dataset later for LSTM or other forecasting models.

### Features
- **Data download** from yfinance for `RELIANCE.NS` (last 5 years)
- **Feature engineering**: SMA20, SMA50, SMA200 on Close prices
- **Clean CSV output** for downstream modeling

### Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install -r requirement.txt
```

If you prefer a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirement.txt
```

### Project Structure
- `DATA.py`: Script to download data and create SMA features
- `Reliance_with_SMA.csv`: Generated dataset with SMA features (output of `DATA.py`)
- `requirement.txt`: Python dependencies

### How to Run
Run:

```bash
python DATA.py
```

Notes:
- `Reliance_with_SMA.csv` will be created in the project root after a successful run.
- If you face network timeouts from yfinance, simply re-run. You can also try a shorter period (e.g., change `period="5y"` to `period="2y"` in `DATA.py`) or ensure a stable connection.

### Next Steps (Optional)
- Use the generated CSV to train an LSTM model
- Add scaling, train/test split, and evaluation scripts
- Experiment with additional technical indicators

### Disclaimer
For educational purposes only; not financial advice.

