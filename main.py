import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Stock symbols
symbols = ['AAPL', 'MSFT', 'NVDA']

# Create directory for storing data if it doesn't exist
data_dir = Path('stock_data')
data_dir.mkdir(exist_ok=True)

# Step 1: Data Collection
# Fetch stock price data
stock_data = yf.download(symbols, start='2020-01-01', end='2023-01-01')

# Convert data to DataFrame and save to CSV
csv_path = data_dir / 'stock_data_yf.csv'
stock_data.to_csv(csv_path)

# Step 2: Data Processing and Analysis
# Load stock data
df_stocks = pd.read_csv(csv_path, header=[0, 1], index_col=0)
df_stocks.index = pd.to_datetime(df_stocks.index)

# Check DataFrame columns
print(df_stocks.head())
print(df_stocks.describe())

# Visualization of stock closing prices
plt.figure(figsize=(14, 7))

# Plot stock closing prices
for symbol in symbols:
    plt.plot(df_stocks.index, df_stocks['Close'][symbol], label=f'{symbol} Close Price')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Close Prices')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plot_path = data_dir / 'stocks_comparison.png'
plt.savefig(plot_path)
plt.show()
