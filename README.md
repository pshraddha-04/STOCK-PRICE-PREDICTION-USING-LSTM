# 📈 Stock Price Prediction using LSTM

## 🔎 Project Overview
This project predicts *Microsoft (MSFT) stock prices* using *Long Short-Term Memory (LSTM)* networks.  
Stock prices are sequential in nature; hence, LSTM models are used to capture *time-series dependencies* better than traditional approaches.  

The pipeline includes:  
1. Data collection from Yahoo Finance  
2. Feature engineering (SMA, RSI, Bollinger Bands, MACD, etc.)  
3. Sequence preparation for LSTM  
4. Baseline model (Linear Regression) for benchmarking  
5. LSTM training and hyperparameter tuning  
6. Evaluation and visualization of predictions 
---
## ⚙ Requirements
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
## 🛠 Tech Stack

**Data Handling:**  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) &nbsp; ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Data Acquisition:**  
![Yahoo Finance](https://img.shields.io/badge/yfinance-400090?style=for-the-badge&logo=yahoo&logoColor=white)

**Visualization:**  
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

**Model Development & Evaluation:**  
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) &nbsp; ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Deployment / Utilities:**  
![Joblib](https://img.shields.io/badge/Joblib-0078D7?style=for-the-badge&logo=python&logoColor=white)

 
---

## 🧠 Why LSTM?
 
- Stock prices form a **time series** → values depend on previous observations.
- **Traditional models (e.g., Simple regression)** assumes independence between features → not suitable for sequential dependencies.
- **LSTM**s are designed to capture **long-term dependencies** in sequential data.
- They handle **lag effects, seasonality, and patterns** better than shallow models.


---

## 📊 Evaluation Metrics
We evaluate models using:  

- **RMSE** – Easier to interpret since in stock price units.  
- **MAE** – Average error magnitude, robust to outliers.  
- **R² Score** – Measures explanatory power (closer to 1 = better).  

✅Using multiple metrics ensures robust evaluation, since stock price prediction is sensitive to both magnitude and variance.

---

## ⚙ Setup
1. Clone repo:
   ```bash
   git clone https://github.com/your-username/Stock-Price-Prediction-LSTM.git
   cd Stock-Price-Prediction-LSTM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---
## 🛠 Workflow and How to Run

The project contains the following Python scripts. The recommended sequence for execution is given as below:

**Workflow:**
 
1. `msft_data_prep.py` → Prepares Microsoft stock data.  
2. `technical_indicators.py` → Computes technical indicators such as SMA, RSI, Bollinger Bands, and MACD.  
3. `plot_indicators.py` → Generates visualizations of the computed technical indicators. 
4. `data_split.py` → Splits the raw dataset into training and testing sets.  
5. `baseline_model.py` → Trains a Linear Regression model as a baseline benchmark.  
6. `feature_selection.py` → Performs feature correlation analysis and selects relevant features.  
7. `prepare_sequences.py` → Scales features and creates input sequences suitable for LSTM models.  
8. `train_initial_lstm.py` → Trains the initial LSTM model on prepared sequences.  
9. `random_search_lstm.py` → Performs Random Search to tune hyperparameters for LSTM models.  
10. `predict_lstm.py` → Loads the trained LSTM model to generate predictions on test data.  
11. `plot_predictions.py` → Plots actual vs predicted stock prices for visual evaluation.  

---

## 📝 Method Highlights
- **Feature Scaling:** Each feature is scaled independently using MinMaxScaler to maintain the relative importance and range of individual indicators.
- **Lag Features:** Past observations (lookback window) are used as inputs to capture temporal dependencies in stock prices.
- **Baseline Model:** Linear Regression provides a performance benchmark before applying advanced sequential models.
- **Hyperparameter Tuning:** Random Search explores different configurations (units, dropout, learning rate, lookback window) to reduce prediction error and optimize model performance.

---
## 🏆 Results

Final LSTM Model Performance on Test Data:

- **Root Mean Squared Error (RMSE):** 8.131  
- **Mean Absolute Error (MAE):** 5.901  
- **R² Score (Coefficient of Determination):** 0.9710  
  Indicates that 97.1% of the variance in the test data is explained by the model, demonstrating excellent predictive capability.
> **Note:** These metrics reflect the performance of the optimized LSTM model after hyperparameter tuning.

---
## 🏁 Conclusion

- This project demonstrates a systematic approach to stock price prediction using LSTM models.  
- Through feature engineering, sequence preparation, and hyperparameter tuning, the final LSTM model achieves strong predictive performance, outperforming baseline models.  
- The framework can be extended to other stocks or financial time-series datasets for further analysis and research.
