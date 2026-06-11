import sqlite3
import pandas as pd

# Connect to database
# This creates a file called finsight.db in your project folder
conn = sqlite3.connect("finsight.db")
print("Connected to database finsight.db")

# Load clean data
credit = pd.read_csv("data/processed/credit_clean.csv")
stocks = pd.read_csv("data/processed/stocks_clean.csv")

# Load credit data into database as a table
credit.to_sql("credit_risk", conn, if_exists="replace", index=False)
print(f"Loaded {len(credit)} rows into 'credit_risk' table")

# Load stock data into database as a table
stocks.to_sql("stock_prices", conn, if_exists="replace", index=False)
print(f"Loaded {len(stocks)} rows into 'stock_prices' table")

# Verify by querying the database
print()
print("Verifying tables in database:")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables)

conn.close()
print()
print("Database built successfully!")