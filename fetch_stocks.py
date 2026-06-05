import yfinance as yf
import pandas as pd
import os

# Create a folder to store our data
os.makedirs("data/raw", exist_ok=True)

companies = {
    "AAPL": "Apple",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "V": "Visa",
    "MA": "Mastercard"
}

print("Fetching 1 year of real stock data from Yahoo Finance...")
print()

all_data = []

for ticker, name in companies.items():
    print(f"  Fetching {name}...")
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    data["ticker"] = ticker
    data["company"] = name
    all_data.append(data)

# Combine all companies into one big table
combined = pd.concat(all_data)

# Save as CSV first so you can open it in Excel and see it
combined.to_csv("data/raw/stock_prices.csv")

print()
print("Saved! Let's look at the data:")
print(f"Total rows: {len(combined)}")
print(f"Columns: {list(combined.columns)}")
print()
print(combined.head())