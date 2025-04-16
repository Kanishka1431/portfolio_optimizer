import yfinance as yf
data = yf.download(['AAPL', 'MSFT', 'NVDA'], period='1y')
print(data.head())