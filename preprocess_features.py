import pandas as pd


def load_data(filepath):
    return pd.read_csv(filepath, parse_dates=['Date'])

def compute_daily_return(df):
    df['Daily Return'] = df.groupby('Ticker')['Close'].pct_change()
    return df

def compute_volatility(df, window=10):
    df['Volatility'] = df.groupby('Ticker')['Daily Return'].rolling(window).std().reset_index(0, drop=True)
    return df

def compute_volume_zscore(df, window=20):
    def zscore(series):
        return (series - series.rolling(window).mean()) / series.rolling(window).std()

    df['Volume Z-Score'] = df.groupby('Ticker')['Volume'].apply(zscore).reset_index(level=0, drop=True)
    return df

def save_features(df):
    df.dropna().to_csv('data/features.csv')
