# Moving Average Z Score

![image](https://github.com/user-attachments/assets/d20912fb-b7f0-4b81-86b4-6fdaf7ea8099)

This project provides a Python-based tool for financial analysis that combines the calculation of various Moving Averages (MAs) with Z-Score analysis to detect potential market divergences. It leverages yfinance for fetching historical stock price data and uses popular techniques from quantitative finance to aid in the detection of market trends and reversals.

<h3 align="left">Features</h3>
Calculate Multiple Types of Moving Averages: SMA, WMA, EMA, HMA
Z-Score Calculation: Computes the Z-Score for a given moving average to standardize the data, providing insight into price deviations from the average.
Divergence Detection: Identifies potential bullish and bearish divergences based on Z-Score behavior, which could indicate upcoming market reversals.
Visual Representation: Plots the stock price and corresponding Z-Score, highlighting detected divergences, to facilitate quick analysis.

<h3 align="left">Applications in Financial Analysis</h3>
Trend Identification: Moving averages help in identifying the direction of the trend. By applying Z-Score analysis, traders can better assess when the price is deviating significantly from the trend.
Divergence Detection: Divergences between the Z-Score and price can indicate potential reversals, making this tool valuable for spotting early entry or exit points.
Volatility Measurement: Z-Score also serves as a measure of price volatility relative to the moving average, helping traders to assess the risk level of their positions.
Quantitative Strategy Development: This tool provides a foundation for building more sophisticated trading strategies that can be backtested and optimized for different market conditions.

<h3 align="left">Dependencies</h3>
Python 3.7+
yfinance
pandas
numpy
matplotlib

