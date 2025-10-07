from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_lstm_model.pkl")
SCALER_DIR = os.path.join(BASE_DIR, "data", "scalers")

# Load model and scalers
model = joblib.load(MODEL_PATH)
scalers = {}
feature_names = ["Close", "Open", "Volume", "RSI_14", "MACD", "MACD_Hist"]
for feature in feature_names:
    scaler_path = os.path.join(SCALER_DIR, f"{feature}_scaler.pkl")
    scalers[feature] = joblib.load(scaler_path)

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_hist

def send_email(name, email, subject, message):
    # Email configuration - replace with your email settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "your-email@gmail.com"  # Replace with your email
    sender_password = "your-app-password"   # Replace with your app password
    recipient_email = "recipient@gmail.com"  # Replace with recipient email
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Contact Form: {subject}"
    
    body = f"""
    New contact form submission:
    
    Name: {name}
    Email: {email}
    Subject: {subject}
    
    Message:
    {message}
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()

@app.route("/contact", methods=["POST"])
def contact_form():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        
        if not all([name, email, subject, message]):
            return jsonify({"error": "All fields are required"}), 400
        
        send_email(name, email, subject, message)
        return jsonify({"success": True, "message": "Email sent successfully!"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'stockSymbol' not in data:
            return jsonify({"error": "Stock symbol is required"}), 400
            
        symbol = data['stockSymbol'].upper()
        prediction_days = int(data.get('predictionDays', 1))
        
       
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        if hist.empty:
            return jsonify({"error": f"No data found for symbol {symbol}"}), 400
            
   
        hist['RSI_14'] = calculate_rsi(hist['Close'])
        hist['MACD'], hist['MACD_Hist'] = calculate_macd(hist['Close'])
        hist = hist.dropna()
        
        if len(hist) < 60:
            return jsonify({"error": "Insufficient data for prediction"}), 400
            
        current_price = float(hist['Close'].iloc[-1])
        
        
        sequence = hist[feature_names].iloc[-60:].values
        scaled_sequence = np.zeros_like(sequence)
        for i, feature in enumerate(feature_names):
            scaled_sequence[:, i] = scalers[feature].transform(sequence[:, [i]]).flatten()
        
        X_input = scaled_sequence.reshape(1, 60, len(feature_names))
        pred_scaled = model.predict(X_input, verbose=0)
        next_day_prediction = float(scalers['Close'].inverse_transform(pred_scaled)[0][0])
        
       
        daily_change_pct = (next_day_prediction - current_price) / current_price
        
       
        historical_volatility = hist['Close'].pct_change().std() * np.sqrt(252)
        
       
        if prediction_days == 1:
            final_prediction = next_day_prediction
        else:
            
            time_scaling = np.sqrt(prediction_days)
            scaled_change = daily_change_pct * time_scaling
            
        
            uncertainty = historical_volatility * np.sqrt(prediction_days / 252) * 0.3
            random_factor = np.random.normal(0, uncertainty)
            scaled_change += random_factor
            
            
            max_change = historical_volatility * np.sqrt(prediction_days / 252) * 2
            scaled_change = np.clip(scaled_change, -max_change, max_change)
            
            final_prediction = current_price * (1 + scaled_change)
        price_change = float(final_prediction - current_price)
        percent_change = float((price_change / current_price) * 100)
        
        return jsonify({
            "success": True,
            "symbol": symbol,
            "prediction_days": prediction_days,
            "current_price": round(current_price, 2),
            "predicted_price": round(final_prediction, 2),
            "price_change": round(price_change, 2),
            "percent_change": round(percent_change, 2),
            "confidence": 97.1,
            "rmse": 8.131,
            "mae": 5.901,
            "r2_score": 0.9710
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return send_from_directory('../web', 'dashboard.html')

@app.route('/index.html')
def index():
    return send_from_directory('../web', 'index.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('../web', 'dashboard.html')

@app.route('/about.html')
def about():
    return send_from_directory('../web', 'about.html')

@app.route('/contact.html')
def contact_page():
    return send_from_directory('../web', 'contact.html')

@app.route('/features.html')
def features():
    return send_from_directory('../web', 'features.html')

@app.route('/results.html')
def results():
    return send_from_directory('../web', 'results.html')

@app.route('/css/<path:filename>')
def css_files(filename):
    return send_from_directory('../web/css', filename)

@app.route('/js/<path:filename>')
def js_files(filename):
    return send_from_directory('../web/js', filename)

if __name__ == "__main__":
    app.run(debug=True)