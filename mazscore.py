import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_price_data(ticker, period="2y"):
    data = yf.download(ticker, period=period)
    data['Close'] = data['Adj Close']
    return data

def hull_moving_average(data, period):
    half_period_wma = data.rolling(window=period//2).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    full_period_wma = data.rolling(window=period).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    raw_hma = 2 * half_period_wma - full_period_wma
    hma = raw_hma.rolling(window=int(np.sqrt(period))).mean()
    return hma

def calculate_ma(data, period, ma_type="SMA"):
    if ma_type == "SMA":
        return data['Close'].rolling(window=period).mean()
    elif ma_type == "Hull":
        return hull_moving_average(data['Close'], period)
    elif ma_type == "Ema":
        return data['Close'].ewm(span=period, adjust=False).mean()
    elif ma_type == "Wma":
        weights = np.arange(1, period + 1)
        return data['Close'].rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")

def calculate_z_score(data, lookback_period):
    mean = data.rolling(window=lookback_period).mean()
    std_dev = data.rolling(window=lookback_period).std()
    z_score = (data - mean) / std_dev
    return z_score

def detect_divergences(data, lbR, lbL):
    divergences = {"bullish": [], "bearish": []}

    for i in range(lbL, len(data) - lbR):
        low_cond = data[i] < data[i - lbL:i].min()
        high_cond = data[i] > data[i - lbL:i].max()

        if low_cond and data[i] > data[i + 1:i + 1 + lbR].max():
            divergences["bullish"].append((i, data[i]))
        elif high_cond and data[i] < data[i + 1:i + 1 + lbR].min():
            divergences["bearish"].append((i, data[i]))

    return divergences

def moving_average_z_score_suite(ticker, ma_type="SMA", period=30, z_score_lookback=30):
    data = fetch_price_data(ticker)
    ma = calculate_ma(data, period, ma_type)
    z_score = calculate_z_score(ma, z_score_lookback)
    divergences = detect_divergences(z_score, lbR=10, lbL=10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data.index, data['Close'], label="Stock Price", color="black")
    ax1.set_yscale('log')
    ax1.set_title(f"{ticker} - Stock Price")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax2.bar(data.index, z_score, label="Z-Score", color=np.where(z_score > 0, 'green', 'red'))
    ax2.axhline(0, color='black', linewidth=1, linestyle='-')
    
    z_min = z_score.min()
    z_max = z_score.max()

    for div in divergences["bullish"]:
        ax2.plot(data.index[div[0]], z_min, 'ro', label='Bullish Divergence')
    for div in divergences["bearish"]:
        ax2.plot(data.index[div[0]], z_max, 'go', label='Bearish Divergence')

    ax2.set_title("Moving Average Z-Score Suite")
    ax2.set_ylabel("Z-Score")
    ax2.legend()
    
    plt.show()

#RUN

moving_average_z_score_suite(ticker="BTC-USD", ma_type="Hull", period=50, z_score_lookback=30)
