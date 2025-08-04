import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid TclError
import matplotlib.pyplot as plt
import numpy as np

# --- Load anomaly results ---
sliding = pd.read_csv("../output/sliding_anomalies.csv", parse_dates=["Date"], dayfirst=False)
heap = pd.read_csv("../output/heap_anomalies.csv", parse_dates=["Date"], dayfirst=False)

# --- Compare anomalies ---
print(f"Sliding total: {len(sliding)}")
print(f"Heap total: {len(heap)}")

overlap = pd.merge(sliding, heap, on=["Date", "Ticker"])
print(f"Anomalies flagged by both: {len(overlap)}")

# --- Save summary to file ---
with open("anomaly_summary.txt", "w") as f:
    f.write(f"Sliding total: {len(sliding)}\n")
    f.write(f"Heap total: {len(heap)}\n")
    f.write(f"Overlap: {len(overlap)}\n")

# --- Time series plot function ---
def plot_time_series_with_anomalies(anomalies, ticker, method):
    df = pd.read_csv("../data/features.csv", parse_dates=["Date"])
    df = df[df["Ticker"] == ticker]
    anomalies = anomalies[anomalies["Ticker"] == ticker]

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Price"], label="Price", alpha=0.6)
    plt.scatter(anomalies["Date"], anomalies["Price"], color="red", label="Anomalies", s=20)
    plt.title(f"{ticker} - {method} Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_{method}_time_series.png")

# --- Generate time series plots ---
plot_time_series_with_anomalies(sliding, "AAPL", "Sliding Window")
plot_time_series_with_anomalies(heap, "AAPL", "Heap-Based")

# --- Cluster scatter plot (optional) ---
def plot_cluster_scatter(X, labels, method):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", s=10)
    plt.title(f"Cluster Scatter Plot - {method}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(f"{method}_cluster_scatter.png")

# --- Load PCA features and labels ---
try:
    X = np.loadtxt("../data/pca_features.csv", delimiter=",")
    labels = np.loadtxt("../output/sliding_labels.csv", delimiter=",")
    plot_cluster_scatter(X, labels, "Sliding Window")
except Exception as e:
    print("⚠️ Cluster scatter plot skipped:", e)

print("✅ Analysis complete. Plots and summary saved.")
