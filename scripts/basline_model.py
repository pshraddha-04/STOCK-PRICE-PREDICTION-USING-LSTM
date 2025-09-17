import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    return df

def create_lag_features(df: pd.DataFrame, target_col: str = 'Close', lags: int = 5) -> pd.DataFrame:
    for i in range(1, lags + 1):
        df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
    for col in ['SMA_20', 'SMA_50', 'RSI_14', 'MACD']:
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
    return df

def linear_regression_baseline(df: pd.DataFrame, target_col: str = 'Close', test_size: float = 0.2):
    feature_cols = [col for col in df.columns if 'lag' in col]
    X = df[feature_cols]
    y = df[target_col]
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_indices]
    y = y[valid_indices]
    dates = df.index[valid_indices]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    return predictions, y_test, dates_test, model


def calculate_metrics(actual, predicted, model_name: str):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    print(f"\n{model_name} Performance Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    return {'RMSE': rmse, 'MAE': mae}


def plot_predictions(actual, predicted, dates, model_name: str):
    plt.figure(figsize=(12, 7))
    plt.plot(dates, actual, label='Actual Price', linewidth=2, alpha=0.8)
    plt.plot(dates, predicted, label=f'{model_name} Prediction', linewidth=2, alpha=0.8)
    plt.title(f'Stock Price Prediction: {model_name} Baseline Model', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Walk-forward validation (lag-only features)
def create_lag_features_wf(df: pd.DataFrame, target_col: str = "Close", n_lags: int = 5) -> pd.DataFrame:
    df_lag = df.copy()
    for lag in range(1, n_lags + 1):
        df_lag[f"Lag_{lag}"] = df_lag[target_col].shift(lag)
    df_lag.dropna(inplace=True)
    return df_lag


def walk_forward_validation(df: pd.DataFrame, target_col: str = "Close", n_lags: int = 5, model_type: str = "linear"):
    df_lag = create_lag_features_wf(df, target_col, n_lags)
    X = df_lag[[f"Lag_{i}" for i in range(1, n_lags + 1)]].values
    y = df_lag[target_col].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    predictions, actuals = [], []
    for i in range(n_lags, len(X)):
        X_train, y_train = X[:i], y[:i]
        X_test, y_test = X[i].reshape(1, -1), y[i]
        model = LinearRegression() if model_type == "linear" else Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(y_pred[0])
        actuals.append(y_test)
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    mae = float(mean_absolute_error(actuals, predictions))
    return rmse, mae


def main():
    print("--- Linear Regression Baseline ---")

    data_path = "data/microsoft_data_SMA_RSI_BBands_MACD.csv"
    df = load_and_prepare_data(data_path)

    # Linear regression baseline with technical indicator lags
    df_with_lags = create_lag_features(df.copy())
    linear_pred, linear_actual, linear_dates, linear_model = linear_regression_baseline(df_with_lags)
    linear_metrics = calculate_metrics(linear_actual, linear_pred, "Linear Regression")

    # Feature importance 
    feature_cols = [col for col in df_with_lags.columns if 'lag' in col]
    feature_importance = pd.DataFrame({'Feature': feature_cols, 'Coefficient': linear_model.coef_}).sort_values('Coefficient', key=abs, ascending=False)

    # Visualization
    plot_predictions(linear_actual, linear_pred, linear_dates, "Linear Regression")

    # Walk-forward experiments
    print("Walk-forward validation (lag-only features)")
    for lags in [1, 5, 10]:
        for model_type in ["linear", "ridge"]:
            rmse, mae = walk_forward_validation(df[["Close"]], target_col="Close", n_lags=lags, model_type=model_type)
            print(f"Lags={lags:2d}, Model={model_type:<6} => RMSE=${rmse:.2f}, MAE=${mae:.2f}")

    return {
        'linear_metrics': linear_metrics,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    results = main()


