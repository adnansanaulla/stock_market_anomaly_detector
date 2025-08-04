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
print("Loading anomaly data...")
sliding_raw = safe_read_csv("../output/sliding_anomalies.csv", parse_dates=True)
heap_raw = safe_read_csv("../output/heap_anomalies.csv", parse_dates=True)

# --- Filter actual anomalies ---
sliding = sliding_raw[sliding_raw["Anomaly"] == 1].copy() if not sliding_raw.empty else pd.DataFrame()
heap = heap_raw[heap_raw["Anomaly"] == 1].copy() if not heap_raw.empty else pd.DataFrame()

print(f"🔍 Sliding anomalies: {len(sliding)}")
print(f"🔍 Heap anomalies: {len(heap)}")

# --- Load full feature data with real dates ---
print("Loading feature data...")
features = safe_read_csv("../data/features.csv", parse_dates=True)

if features.empty:
    print("❌ No feature data found. Cannot proceed with analysis.")
    exit(1)

print(f"📊 Feature data loaded: {len(features)} rows")

# --- Clean and validate anomaly data ---
def clean_anomaly_data(anomalies, name):
    if anomalies.empty:
        print(f"⚠️ No {name} anomalies to clean.")
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_cols = ['Date', 'Ticker', 'Close']
    missing_cols = [col for col in required_cols if col not in anomalies.columns]
    if missing_cols:
        print(f"❌ Missing columns in {name} anomalies: {missing_cols}")
        return pd.DataFrame()
    
    # Remove rows with invalid data
    anomalies = anomalies.dropna(subset=['Date', 'Close', 'Ticker'])
    anomalies = anomalies[anomalies['Close'].apply(lambda x: isinstance(x, (int, float, np.number)))]
    
    print(f"✅ Cleaned {name} anomalies: {len(anomalies)} valid records")
    return anomalies

sliding = clean_anomaly_data(sliding, "sliding")
heap = clean_anomaly_data(heap, "heap")

# --- Compare overlap ---
if not sliding.empty and not heap.empty:
    # Merge on Date and Ticker to find overlapping anomalies
    overlap = pd.merge(sliding[['Date', 'Ticker', 'Close']], 
                      heap[['Date', 'Ticker', 'Close']], 
                      on=['Date', 'Ticker'], 
                      suffixes=('_sliding', '_heap'))
    print(f"🔍 Overlap anomalies: {len(overlap)}")
else:
    overlap = pd.DataFrame()
    print("⚠️ Skipping overlap comparison due to missing data.")

# --- Write summary file ---
print("Writing summary file...")
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

print("✅ Summary written to anomaly_summary.txt")

# --- Enhanced plot function ---
def plot_time_series_with_anomalies(anomalies, ticker, method):
    # Get feature data for this ticker
    df = features[features["Ticker"] == ticker].sort_values("Date").copy()
    ticker_anomalies = anomalies[anomalies["Ticker"] == ticker].copy()

    if df.empty:
        print(f"⚠️ No feature data found for ticker: {ticker}")
        return
    
    if ticker_anomalies.empty:
        print(f"⚠️ No {method} anomalies for ticker: {ticker}")
        return

    # Ensure dates are datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    ticker_anomalies['Date'] = pd.to_datetime(ticker_anomalies['Date'])
    
    try:
        plt.figure(figsize=(14, 8))
        
        # Plot the full time series
        plt.plot(df["Date"], df["Close"], 
                label="Close Price", alpha=0.7, linewidth=1, color='blue')
        
        # Plot anomalies
        plt.scatter(ticker_anomalies["Date"], ticker_anomalies["Close"], 
                   color="red", label=f"{method} Anomalies", s=50, alpha=0.8, zorder=5)
        
        plt.title(f"{ticker} - {method} Anomaly Detection", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Close Price ($)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Format x-axis for better date display
        plt.xticks(rotation=45)
        
        filename = f"plots/{ticker}_{method}_anomalies.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved plot: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to create plot for {ticker} ({method}): {e}")
        plt.close()

# --- Create overview plots ---
def create_overview_plots():
    try:
        # Plot 1: Anomaly counts by method
        plt.figure(figsize=(10, 6))
        methods = ['Sliding Window', 'Heap-based']
        counts = [len(sliding), len(heap)]
        colors = ['skyblue', 'lightcoral']
        
        bars = plt.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black')
        plt.title('Total Anomalies Detected by Method', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Anomalies', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/anomaly_counts_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved overview plot: plots/anomaly_counts_overview.png")
        
        # Plot 2: Top tickers with most anomalies (sliding window)
        if not sliding.empty:
            plt.figure(figsize=(12, 6))
            top_tickers = sliding['Ticker'].value_counts().head(10)
            
            plt.bar(range(len(top_tickers)), top_tickers.values, 
                   color='skyblue', alpha=0.7, edgecolor='black')
            plt.title('Top 10 Tickers by Sliding Window Anomalies', fontsize=16, fontweight='bold')
            plt.xlabel('Ticker', fontsize=12)
            plt.ylabel('Number of Anomalies', fontsize=12)
            plt.xticks(range(len(top_tickers)), top_tickers.index, rotation=45)
            
            # Add value labels
            for i, v in enumerate(top_tickers.values):
                plt.text(i, v + max(top_tickers.values)*0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('plots/top_tickers_sliding.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved overview plot: plots/top_tickers_sliding.png")
            
    except Exception as e:
        print(f"❌ Failed to create overview plots: {e}")

# --- Generate all plots ---
print("Creating overview plots...")
create_overview_plots()

print("Creating individual ticker plots...")

# Plot sliding window anomalies
if not sliding.empty:
    tickers_to_plot = sliding["Ticker"].value_counts().head(5).index  # Top 5 tickers
    for ticker in tickers_to_plot:
        plot_time_series_with_anomalies(sliding, ticker, "Sliding_Window")

# Plot heap anomalies
if not heap.empty:
    tickers_to_plot = heap["Ticker"].value_counts().head(5).index  # Top 5 tickers
    for ticker in tickers_to_plot:
        plot_time_series_with_anomalies(heap, ticker, "Heap_Based")

# --- Create comparison plot for overlapping anomalies ---
if not overlap.empty:
    try:
        plt.figure(figsize=(10, 6))
        overlap_counts = overlap.groupby('Ticker').size().sort_values(ascending=False).head(10)
        
        plt.bar(range(len(overlap_counts)), overlap_counts.values, 
               color='purple', alpha=0.7, edgecolor='black')
        plt.title('Overlapping Anomalies by Ticker', fontsize=16, fontweight='bold')
        plt.xlabel('Ticker', fontsize=12)
        plt.ylabel('Number of Overlapping Anomalies', fontsize=12)
        plt.xticks(range(len(overlap_counts)), overlap_counts.index, rotation=45)
        
        # Add value labels
        for i, v in enumerate(overlap_counts.values):
            plt.text(i, v + max(overlap_counts.values)*0.01, str(v), 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/overlapping_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved overlap plot: plots/overlapping_anomalies.png")
        
    except Exception as e:
        print(f"❌ Failed to create overlap plot: {e}")

print("\n" + "="*50)
print("📊 ANALYSIS COMPLETE")
print("="*50)
print(f"📈 Total sliding window anomalies: {len(sliding)}")
print(f"📈 Total heap-based anomalies: {len(heap)}")
print(f"🔄 Overlapping anomalies: {len(overlap)}")
print(f"📁 Plots saved in: plots/ directory")
print(f"📄 Summary saved in: anomaly_summary.txt")
print("="*50)