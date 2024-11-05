# 📈 Moving Average Z-Score Analysis Suite

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)

A Python tool for technical analysis using Moving Average Z-Score with divergence detection.

## 🚀 Features

- 📊 Multiple Moving Average types (SMA, Hull, EMA, WMA)
- 📉 Z-Score calculation and analysis
- 🎯 Automatic divergence detection
- 📈 Interactive visualization
- 💹 Support for stocks and cryptocurrencies

## 📋 Prerequisites

Before running the script, ensure you have Python 3.7+ installed on your system.

## 🔧 Installation

1. Clone the repository
```bash
git clone https://github.com/YavuzAkbay/moving-average-z-score.git
cd ma-zscore-analysis
```

2. Install required packages
```bash
pip install -r requirements.txt
```

## 💻 Usage

Basic usage:
```python
from ma_zscore_analysis import MAZScoreAnalyzer

# Create analyzer instance
analyzer = MAZScoreAnalyzer(
    ticker="BTC-USD",
    ma_type="Hull",
    period=50,
    z_score_lookback=30
)

# Run analysis
analyzer.run_analysis()
```

Advanced usage:
```python
analyzer = MAZScoreAnalyzer(ticker="AAPL")

# Customize analysis steps
analyzer.fetch_price_data(period="1y")
analyzer.calculate_ma()
analyzer.calculate_z_score()
analyzer.detect_divergences(lbR=15, lbL=15)
analyzer.plot_analysis()
plt.show()
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| ticker | Trading symbol | Required |
| ma_type | Moving average type (SMA/Hull/Ema/Wma) | "SMA" |
| period | MA calculation period | 30 |
| z_score_lookback | Z-Score lookback period | 30 |

## 📊 Output

The analysis generates a two-panel plot:
1. Price chart with moving average (log scale)
2. Z-Score histogram with divergence indicators

## 🔍 Technical Details

### Moving Average Types
- **SMA**: Simple Moving Average
- **Hull**: Hull Moving Average
- **EMA**: Exponential Moving Average
- **WMA**: Weighted Moving Average

### Z-Score Calculation
The Z-Score is calculated as:
```python
Z = (Value - Mean) / Standard_Deviation
```

### Divergence Detection
Detects both bullish and bearish divergences using local extrema comparison.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- `yfinance` for market data access
- Technical Analysis community for insights

## 📧 Contact

Your Name - [akbay.yavuz@gmail.com](mailto:akbay.yavuz@gmail.com)

Project Link: [https://github.com/YavuzAkbay/K-Means-Clustering](https://github.com/YavuzAkbay/K-Means-Clustering)

---
⭐️ If you found this project helpful, please consider giving it a star!
