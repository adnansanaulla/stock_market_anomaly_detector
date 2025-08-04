import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# --- Load CSV safely ---
def safe_read_csv(filepath, parse_dates=False):
    try:
        df = pd.read_csv(filepath)
        if parse_dates and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df[df["Date"].notna()]
        return df
    except Exception as e:
        print(f"⚠️ Failed to read {filepath}: {e}")
        return pd.DataFrame()

# --- Load raw anomaly data ---
sliding_raw = safe_read_csv("../output/sliding_anomalies.csv")
heap_raw = safe_read_csv("../output/heap_anomalies.csv")

# --- Filter actual anomalies ---
sliding = sliding_raw[sliding_raw["Anomaly"] == 1].copy()
heap = heap_raw[heap_raw["Anomaly"] == 1].copy()

print(f"🔍 Sliding anomalies: {len(sliding)}")
print(f"🔍 Heap anomalies: {len(heap)}")

# --- Load full feature data with real dates ---
features = safe_read_csv("../data/features.csv", parse_dates=True)

# --- Replace fake Date index with actual Date from features.csv ---
def inject_real_dates(anomalies, features):
    if anomalies.empty:
        print("⚠️ No anomalies to process.")
        return pd.DataFrame()

    fixed = []
    for ticker in anomalies["Ticker"].unique():
        df_anom = anomalies[anomalies["Ticker"] == ticker].copy()
        df_feat = features[features["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        if df_feat.empty:
            print(f"⚠️ No feature data for ticker: {ticker}")
            continue

        df_anom["RealDate"] = df_anom["Date"].astype(int).apply(
            lambda idx: df_feat.iloc[idx]["Date"] if idx < len(df_feat) else pd.NaT
        )
        fixed.append(df_anom)

    if not fixed:
        print("⚠️ No valid ticker matches found. Skipping date injection.")
        return pd.DataFrame()

    return pd.concat(fixed).dropna(subset=["RealDate"]).rename(columns={"RealDate": "Date"})

# --- Apply date fix ---
sliding = inject_real_dates(sliding, features)
heap = inject_real_dates(heap, features)

# --- Compare overlap ---
if not sliding.empty and not heap.empty:
    overlap = pd.merge(sliding, heap, on=["Date", "Ticker"])
    print(f"🔍 Overlap anomalies: {len(overlap)}")
else:
    overlap = pd.DataFrame()
    print("⚠️ Skipping overlap comparison due to missing data.")

# --- Write summary file ---
with open("anomaly_summary.txt", "w") as f:
    f.write(f"Sliding total: {len(sliding)}\n")
    f.write(f"Heap total: {len(heap)}\n")
    f.write(f"Overlap: {len(overlap)}\n\n")

    f.write("Sliding anomalies per ticker:\n")
    if not sliding.empty and "Ticker" in sliding.columns:
        for ticker, count in sliding["Ticker"].value_counts().items():
            f.write(f"{ticker}: {count}\n")
    else:
        f.write("No sliding anomalies found.\n")

    f.write("\nHeap anomalies per ticker:\n")
    if not heap.empty and "Ticker" in heap.columns:
        for ticker, count in heap["Ticker"].value_counts().items():
            f.write(f"{ticker}: {count}\n")
    else:
        f.write("No heap anomalies found.\n")

# --- Plot function ---
def plot_time_series_with_anomalies(anomalies, ticker, method):
    df = features[features["Ticker"] == ticker].sort_values("Date")
    anomalies = anomalies[anomalies["Ticker"] == ticker]

    if df.empty:
        print(f"⚠️ No data found for ticker: {ticker}")
        return
    if anomalies.empty:
        print(f"⚠️ No anomalies for ticker: {ticker} using {method}")
        return

    anomalies = anomalies.dropna(subset=["Date", "Close"])
    anomalies = anomalies[anomalies["Close"].apply(lambda x: isinstance(x, (int, float, np.number)))]

    x = anomalies["Date"]
    y = anomalies["Close"]

    if len(x) != len(y):
        print(f"❌ Skipping plot for {ticker}: Date and Close mismatch ({len(x)} vs {len(y)})")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Close"], label="Close Price", alpha=0.6)
    plt.scatter(x, y, color="red", label="Anomalies", s=20)
    plt.title(f"{ticker} - {method} Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    filename = f"plots/{ticker}_{method}_time_series.png"
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")

# --- Plot tickers with anomalies ---
for ticker in sliding["Ticker"].unique():
    plot_time_series_with_anomalies(sliding, ticker, "Sliding")

for ticker in heap["Ticker"].unique():
    plot_time_series_with_anomalies(heap, ticker, "Heap")

# --- Optional cluster plot ---
def plot_cluster_scatter(X, labels, method):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", s=10)
    plt.title(f"Cluster Scatter Plot - {method}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    filename = f"plots/{method}_cluster_scatter.png"
    plt.savefig(filename)
    print(f"✅ Saved cluster plot: {filename}")

# --- Load PCA features and labels ---
if os.path.exists("../data/pca_features.csv") and os.path.exists("../output/sliding_labels.csv"):
    try:
        X = np.loadtxt("../data/pca_features.csv", delimiter=",")
        labels = np.loadtxt("../output/sliding_labels.csv", delimiter=",")
        plot_cluster_scatter(X, labels, "Sliding")
    except Exception as e:
        print("⚠️ Cluster plot skipped due to error:", e)
else:
    print("⚠️ PCA features or labels missing. Skipping cluster plot.")

print("✅ All done. Summary and plots saved.")
