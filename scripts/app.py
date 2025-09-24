# app.py

from flask import Flask, request, jsonify

app = Flask(__name__)

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON input from request
    data = request.get_json()

    # Extract fields (optional for now, just to show structure)
    stock = data.get("stock", "N/A")
    days = data.get("days", "N/A")

    # Return dummy response
    return jsonify({
        "stock": "AAPL",
        "days": 5,
        "prediction": "test"
    })

if __name__ == "__main__":
    app.run(debug=True)