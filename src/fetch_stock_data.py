import yfinance as yf
import pandas as pd

# 50 tickers
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'BAC',
    'V', 'MA', 'UNH', 'HD', 'PG', 'DIS', 'PEP', 'KO', 'XOM', 'CVX',
    'ADBE', 'CRM', 'CSCO', 'ORCL', 'INTC', 'T', 'WMT', 'BA', 'NKE', 'PFE',
    'MRK', 'LLY', 'ABBV', 'MCD', 'IBM', 'GE', 'F', 'GM', 'QCOM', 'UPS',
    'COST', 'SBUX', 'GS', 'DE', 'CAT', 'LMT', 'RTX', 'BKNG', 'BLK', 'MMM']

dfs = []
# loading all the data
for ticker in tickers:
    df = yf.download(ticker, start='2010-01-01', end='2025-01-01', progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Ticker'] = ticker
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.sort_values(['Date', 'Ticker']).reset_index(drop=True)
combined.to_csv('data/stock_data.csv', index=False)

# data format is Index,Date,Open,High,Low,Close,Adj Close,Volume,Ticker
# valid dates until 184154