import pandas as pd

# Load CSV file given by your teammates
df = pd.read_csv("processed_stock_clusters.csv", parse_dates=["Date"])


import matplotlib.pyplot as plt

def plot_time_series(df, ticker, method='kmeans'):
    stock_df = df[df['Ticker'] == ticker].sort_values('Date')
    anomalies = stock_df[stock_df[f'is_{method}_anomaly'] == 1]

    plt.figure(figsize=(12, 5))
    plt.plot(stock_df['Date'], stock_df['DailyReturn'], label='Daily Return')
    plt.scatter(anomalies['Date'], anomalies['DailyReturn'], color='red', label='Anomaly', marker='x')
    plt.title(f"{ticker} - {method.upper()} Detected Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_cluster_scatter(df, ticker, method='kmeans'):
    stock_df = df[df['Ticker'] == ticker]
    x = stock_df['DailyReturn']
    y = stock_df['VolumeZScore']
    labels = stock_df[f'{method}_cluster']

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel("Daily Return")
    plt.ylabel("Volume Z-Score")
    plt.title(f"{ticker} - {method.upper()} Clusters")
    plt.colorbar(scatter, label='Cluster')
    plt.grid()
    plt.tight_layout()
    plt.show()


def compare_anomaly_counts(df):
    counts = {
        'KMeans Anomalies': df['is_kmeans_anomaly'].sum(),
        'DBSCAN Anomalies': df['is_dbscan_anomaly'].sum(),
        'Both': ((df['is_kmeans_anomaly'] == 1) & (df['is_dbscan_anomaly'] == 1)).sum()
    }
    return counts
