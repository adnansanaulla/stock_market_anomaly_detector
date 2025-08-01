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
    df = yf.download(ticker, start='2010-01-01', end='2025-01-01')
    df['Ticker'] = ticker
    dfs.append(df.reset_index())

combined = pd.concat(dfs)
combined.to_csv('stock_data.csv')