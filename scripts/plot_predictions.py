import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "test_predictions.csv")

# Load predictions
predictions_df = pd.read_csv(PREDICTIONS_PATH)

# Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(predictions_df["Actual"], label="Actual Price", color="blue")
plt.plot(predictions_df["Predicted"], label="Predicted Price", color="red")
plt.title("LSTM Model: Actual vs Predicted Stock Price (Test Data)")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Save plot as image
plot_path = os.path.join(DATA_DIR, "test_predictions_plot.png")
plt.savefig(plot_path)
print(f" Plot saved to: {plot_path}")