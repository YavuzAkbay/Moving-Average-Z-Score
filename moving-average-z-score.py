import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class MAZScoreAnalyzer:
    """
    A class for analyzing financial instruments using Moving Average Z-Score analysis.
    """
    
    def __init__(self, ticker: str, ma_type: str = "SMA", 
                 period: int = 30, z_score_lookback: int = 30):
        """
        Initialize the analyzer with specified parameters.
        
        Args:
            ticker: Stock/crypto symbol
            ma_type: Type of moving average ("SMA", "Hull", "Ema", "Wma")
            period: Period for moving average calculation
            z_score_lookback: Lookback period for Z-score calculation
        """
        self.ticker = ticker
        self.ma_type = ma_type
        self.period = period
        self.z_score_lookback = z_score_lookback
        self.data = None
        self.ma = None
        self.z_score = None
        self.divergences = None
        
    def fetch_price_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch price data from Yahoo Finance."""
        self.data = yf.download(self.ticker, period=period)
        self.data['Close'] = self.data['Adj Close']
        return self.data

    @staticmethod
    def hull_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        weights = lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        
        half_period_wma = data.rolling(window=period//2).apply(weights, raw=True)
        full_period_wma = data.rolling(window=period).apply(weights, raw=True)
        
        raw_hma = 2 * half_period_wma - full_period_wma
        hma = raw_hma.rolling(window=int(np.sqrt(period))).mean()
        return hma

    def calculate_ma(self) -> pd.Series:
        """Calculate the specified type of moving average."""
        if self.ma_type == "SMA":
            self.ma = self.data['Close'].rolling(window=self.period).mean()
        elif self.ma_type == "Hull":
            self.ma = self.hull_moving_average(self.data['Close'], self.period)
        elif self.ma_type == "Ema":
            self.ma = self.data['Close'].ewm(span=self.period, adjust=False).mean()
        elif self.ma_type == "Wma":
            weights = np.arange(1, self.period + 1)
            self.ma = self.data['Close'].rolling(self.period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        else:
            raise ValueError(f"Unknown MA type: {self.ma_type}")
        return self.ma

    def calculate_z_score(self) -> pd.Series:
        """Calculate Z-score for the moving average."""
        mean = self.ma.rolling(window=self.z_score_lookback).mean()
        std_dev = self.ma.rolling(window=self.z_score_lookback).std()
        self.z_score = (self.ma - mean) / std_dev
        return self.z_score

    def detect_divergences(self, lbR: int = 10, lbL: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """Detect bullish and bearish divergences in Z-score."""
        self.divergences = {"bullish": [], "bearish": []}
        z_score_values = self.z_score.values
        
        for i in range(lbL, len(z_score_values) - lbR):
            if np.isnan(z_score_values[i]):
                continue
                
            low_cond = z_score_values[i] < np.nanmin(z_score_values[i - lbL:i])
            high_cond = z_score_values[i] > np.nanmax(z_score_values[i - lbL:i])
            
            if low_cond and z_score_values[i] > np.nanmax(z_score_values[i + 1:i + 1 + lbR]):
                self.divergences["bullish"].append((i, z_score_values[i]))
            elif high_cond and z_score_values[i] < np.nanmin(z_score_values[i + 1:i + 1 + lbR]):
                self.divergences["bearish"].append((i, z_score_values[i]))
                
        return self.divergences

    def plot_analysis(self):
        """Plot the analysis results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Price plot
        ax1.plot(self.data.index, self.data['Close'], label="Price", color="black")
        ax1.plot(self.data.index, self.ma, label=f"{self.ma_type} ({self.period})", 
                color="blue", alpha=0.7)
        ax1.set_yscale('log')
        ax1.set_title(f"{self.ticker} - Price and {self.ma_type}")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Z-Score plot
        colors = np.where(self.z_score > 0, 'green', 'red')
        ax2.bar(self.data.index, self.z_score, label="Z-Score", color=colors)
        ax2.axhline(0, color='black', linewidth=1, linestyle='-')
        
        # Plot divergences
        z_min, z_max = self.z_score.min(), self.z_score.max()
        for div in self.divergences["bullish"]:
            ax2.plot(self.data.index[div[0]], z_min, 'ro', label='Bullish Divergence')
        for div in self.divergences["bearish"]:
            ax2.plot(self.data.index[div[0]], z_max, 'go', label='Bearish Divergence')
            
        ax2.set_title(f"Moving Average Z-Score (Lookback: {self.z_score_lookback})")
        ax2.set_ylabel("Z-Score")
        ax2.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        return fig

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        self.fetch_price_data()
        self.calculate_ma()
        self.calculate_z_score()
        self.detect_divergences()
        self.plot_analysis()
        plt.show()

if __name__ == "__main__":
    analyzer = MAZScoreAnalyzer(
        ticker="BTC-USD",
        ma_type="Hull",
        period=50,
        z_score_lookback=30
    )
    analyzer.run_analysis()