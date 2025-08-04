import pandas as pd
import matplotlib.pyplot as plt

# Load both result CSVs
sliding = pd.read_csv("output/sliding_anomalies.csv", parse_dates=["Date"])
heap = pd.read_csv("output/heap_anomalies.csv", parse_dates=["Date"])

def plot_anomalies(df, ticker, method):
    df = df[df["Ticker"] == ticker].sort_values("Date")
    anomalies = df[df["Anomaly"] == 1]

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Daily Return"], label="Daily Return")
    plt.scatter(anomalies["Date"], anomalies["Daily Return"], color="red", label="Anomaly", marker='x')
    plt.title(f"{ticker} - {method} Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example use
plot_anomalies(sliding, "AAPL", "Sliding Window")
plot_anomalies(heap, "AAPL", "Heap-Based")

# Count comparison
print("Sliding total:", sliding['Anomaly'].sum())
print("Heap total:", heap['Anomaly'].sum())

merged = sliding.copy()
merged['Heap_Anomaly'] = heap['Anomaly']
merged['Both'] = (merged['Anomaly'] == 1) & (merged['Heap_Anomaly'] == 1)
print("Anomalies flagged by both:", merged['Both'].sum())
