# 📈 Stock Price Prediction using LSTM

This project aims to **predict stock prices using LSTM (Long Short-Term Memory)** deep learning models.  


---

## 🏷️ Project Badges
![Python](https://img.shields.io/badge/Python-3.9+-blue)  
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📂 Project Structure
```bash
STOCK-PRICE-PREDICTION-USING-LSTM/
├── data/
│   ├── microsoft_stock_raw.csv
│   ├── microsoft_stock_clean.csv
│
├── scripts/
│   └── msft_data_prep.py
│
├── requirement.txt
├── .gitignore
└── README.md

```


## ⚙️ Requirements

- Python **3.9+**

---

### 📦 Main Libraries

#### 📊 Data Handling
- 🐼 **pandas** → Data handling & preprocessing  
- 🔢 **numpy** → Numerical computations  

#### 📈 Data Acquisition
- 📈 **yfinance** → Fetch historical stock data  

#### 📊 Visualization
- 📊 **matplotlib** → Visualization  

#### 🤖 Model Development & Evaluation
- 🧠 **tensorflow** → Deep learning model development  
- ⚙️ **scikit-learn** → Model evaluation & preprocessing  

#### 🚀 Deployment
- 🌐 **flask** → Web deployment  
- 💾 **joblib** → Model saving & loading   

---

🚀 How to Run

1.Clone the repository

```bash
git clone https://github.com/<your-username>/STOCK-PRICE-PREDICTION-USING-LSTM.git
cd STOCK-PRICE-PREDICTION-USING-LSTM
```

2.Create a virtual environment

```bash
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3.Install dependencies
```bash
pip install -r requirement.txt
```

4.Run the script to fetch Microsoft stock data
```bash
python scripts/msft_data_prep.py

```
---
## ⚠️ Note

Do not commit your virtual environment (venv/).

It is already included in .gitignore.
