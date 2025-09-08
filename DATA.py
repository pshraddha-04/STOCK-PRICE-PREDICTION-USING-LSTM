import os
import time
import yfinance as yf
import pandas as pd


def download_with_retry(
    ticker: str,
    period: str | None = "5y",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    max_attempts: int = 4,
    base_delay_seconds: int = 2,
    backoff_multiplier: int = 2,
) -> pd.DataFrame:
    """Download ticker data with simple exponential backoff retry.

    Retries on any exception or empty DataFrame result.
    """
    attempt_number = 0
    delay_seconds = base_delay_seconds

    while attempt_number < max_attempts:
        attempt_number += 1
        try:
            # yfinance accepts either period OR start/end
            if start is not None or end is not None:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    threads=False,
                )
            else:
                data = yf.download(
                    ticker,
                    period=period,
                    progress=False,
                    threads=False,
                )

            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
            print(f"Attempt {attempt_number}/{max_attempts}: Empty data received. Retrying in {delay_seconds}s...")
        except Exception as error:
            print(f"Attempt {attempt_number}/{max_attempts} failed: {error}. Retrying in {delay_seconds}s...")

        time.sleep(delay_seconds)
        delay_seconds *= backoff_multiplier

    return pd.DataFrame()


def compute_sma_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA20/50/200 on Close column and drop initial NaNs."""
    data = data.copy()
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()
    return data.dropna()


def save_with_fallback(df: pd.DataFrame, output_path: str) -> None:
    try:
        df.to_csv(output_path)
        print(f"Dataset saved to {output_path} ✅")
    except PermissionError:
        alt_output_path = f"{os.path.splitext(output_path)[0]}_{int(time.time())}.csv"
        df.to_csv(alt_output_path)
        print(f"Target file '{output_path}' is locked. Saved as '{alt_output_path}' instead.")


def ensure_updated_dataset(ticker: str, output_path: str = "Reliance_with_SMA.csv") -> None:
    """Create or update the dataset.

    - If CSV does not exist: download last 5y and create it
    - If CSV exists: download data from (last_date + 1 day) to today, merge, recompute SMAs, and save
    """
    if not os.path.exists(output_path):
        # Fresh download for 5 years
        data = download_with_retry(ticker, period="5y", max_attempts=4)
        if data.empty:
            print("Failed to download data after retries. No file created.")
            return
        data = compute_sma_columns(data)
        save_with_fallback(data, output_path)
        print(data.head())
        return

    # Load existing dataset
    try:
        # Read existing CSV and ensure the index is a datetime index
        existing = pd.read_csv(output_path, index_col=0, parse_dates=[0])
        if not isinstance(existing.index, pd.DatetimeIndex):
            existing.index = pd.to_datetime(existing.index, errors="coerce")
        # Drop any rows with invalid dates
        existing = existing[existing.index.notna()]
    except Exception as error:
        print(f"Failed to read existing CSV '{output_path}': {error}. Recreating from scratch.")
        existing = pd.DataFrame()

    # If existing is empty or malformed, recreate
    if existing.empty or "Close" not in existing.columns:
        data = download_with_retry(ticker, period="5y", max_attempts=4)
        if data.empty:
            print("Failed to download data after retries. No file created.")
            return
        data = compute_sma_columns(data)
        save_with_fallback(data, output_path)
        print(data.head())
        return

    # Determine the last available date and fetch new data since then
    last_date = existing.index.max()
    if not isinstance(last_date, pd.Timestamp):
        try:
            last_date = pd.to_datetime(last_date)
        except Exception:
            print("Could not parse last date from existing file; recreating from scratch.")
            data = download_with_retry(ticker, period="5y", max_attempts=4)
            if data.empty:
                print("Failed to download data after retries. No file created.")
                return
            data = compute_sma_columns(data)
            save_with_fallback(data, output_path)
            print(data.head())
            return

    next_day = (last_date + pd.Timedelta(days=1)).date()
    today = pd.Timestamp.today().date()
    if next_day > today:
        print("Dataset is already up to date.")
        return

    new_data = download_with_retry(ticker, start=str(next_day), end=None, max_attempts=4)

    if new_data.empty:
        print("No new data available or download failed; keeping existing file.")
        return

    # Concatenate and recompute SMAs to ensure continuity
    # Keep only base OHLCV/Adj Close columns from both frames before recomputing SMAs
    base_cols = [col for col in [
        "Open", "High", "Low", "Close", "Adj Close", "Volume"
    ] if col in existing.columns or col in new_data.columns]
    combined = pd.concat([
        existing[base_cols],
        new_data[base_cols]
    ], axis=0)

    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    combined = compute_sma_columns(combined)
    save_with_fallback(combined, output_path)
    print("Dataset updated with latest rows ✅")
    print(combined.tail())


# Run update/create flow
ensure_updated_dataset("RELIANCE.NS", output_path="Reliance_with_SMA.csv")
