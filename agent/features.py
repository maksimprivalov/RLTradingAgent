import numpy as np
import pandas as pd
import yfinance as yf

# load data and return filtered dataframe with additional trading features
def load_data(ticker="MSFT", start="2015-01-01", end="2025-01-01"):
    df = yf.download(tickers=ticker, start=start, end=end)
    
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["sma20"] = df["Close"].rolling(window=20).mean()
    df["sma50"] = df["Close"].rolling(window=50).mean()

    df["volatility20"] = df["log_return"].rolling(window=20).std()
    df["ema12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bollinger_mid"] = df["Close"].rolling(20).mean()
    df["bollinger_std"] = df["Close"].rolling(20).std()

    df["bollinger_up"] = df["bollinger_mid"] + 2 * df["bollinger_std"]
    df["bollinger_down"] = df["bollinger_mid"] - 2 * df["bollinger_std"]
    
    df["volume_change"] = df["Volume"].pct_change()
    
    df = df.reset_index()
    df = df.dropna()

    return df

# splitting for the train, test and validation scopes.
def split_data(df, 
               train_start="2015-01-01", train_end="2020-12-31", 
               test_start="2021-01-01", test_end="2023-12-31",
               val_start="2024-01-01", val_end="2024-12-31"):
    print(df)
    train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)]
    test  = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]
    val   = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)]
    return train, test, val

