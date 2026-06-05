import pandas as pd

# Create a simple table of financial data
data = {
    "company": ["Apple", "JPMorgan", "Goldman Sachs", "Visa", "Mastercard"],
    "stock_price": [189.5, 198.3, 412.7, 275.4, 421.1],
    "revenue_billion": [394, 158, 47, 32, 25],
    "risk_score": [0.12, 0.34, 0.28, 0.15, 0.18]
}

df = pd.DataFrame(data)

print("Our financial data table:")
print(df)
print()
print("Shape of table (rows, columns):", df.shape)
print()
print("Average stock price:", df["stock_price"].mean())