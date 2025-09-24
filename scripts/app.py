# app.py

from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Path to model
MODEL_PATH = os.path.join("models", "best_lstm_model.pkl")

# Load model at startup
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    # For now, ignore features, just confirm model is loaded
    data = request.get_json()

    return jsonify({
        "message": "Model loaded successfully!",
        "input_received": data,
        "prediction": "test"  # placeholder until logic is added
    })

if __name__ == "__main__":
    app.run(debug=True)