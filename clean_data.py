import pandas as pd
import os

# ── Step 1: Load raw data ────────────────────────────────────────────
df = pd.read_csv("data/raw/credit/cs-training.csv")

print("BEFORE cleaning:")
print("Shape:", df.shape)
print()

# ── Step 2: Remove junk column ───────────────────────────────────────
# 'Unnamed: 0' is just a row number - we don't need it
df = df.drop(columns=["Unnamed: 0"])
print("Removed junk column 'Unnamed: 0'")

# ── Step 3: Fix missing values ───────────────────────────────────────
# MonthlyIncome - fill with median (middle value)
# We use median not mean because income data has outliers
median_income = df["MonthlyIncome"].median()
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(median_income)
print(f"Filled missing MonthlyIncome with median: ${median_income:,.0f}")

# NumberOfDependents - fill with mode (most common value)
mode_dependents = df["NumberOfDependents"].mode()[0]
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(mode_dependents)
print(f"Filled missing NumberOfDependents with mode: {mode_dependents}")

# ── Step 4: Verify no missing values remain ──────────────────────────
print()
print("Missing values after cleaning:")
print(df.isnull().sum())

# ── Step 4b: Cap extreme outliers ─────────────────────────────────────
# RevolvingUtilizationOfUnsecuredLines should realistically be 0 to ~2
# Values above 2 are data entry errors, not real customers
# We cap them at 2 rather than deleting rows (preserves the customer record)
before_capping = (df["RevolvingUtilizationOfUnsecuredLines"] > 2).sum()
df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=2)
print(f"Capped {before_capping} extreme outlier values in RevolvingUtilizationOfUnsecuredLines")

# DebtRatio has the same issue - check and cap similarly
before_capping_debt = (df["DebtRatio"] > 5).sum()
df["DebtRatio"] = df["DebtRatio"].clip(upper=5)
print(f"Capped {before_capping_debt} extreme outlier values in DebtRatio")

# ── Step 5: Save clean data ──────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/credit_clean.csv", index=False)

print()
print("AFTER cleaning:")
print("Shape:", df.shape)
print()
print("Clean data saved to data/processed/credit_clean.csv")

# ── Step 6: Clean stock data ─────────────────────────────────────────
df_stocks = pd.read_csv("data/raw/stock_prices.csv")

print()
print("Stock data before cleaning:")
print("Shape:", df_stocks.shape)
print("Missing values:", df_stocks.isnull().sum().sum())

# Reset index and rename date column
df_stocks = df_stocks.reset_index()
df_stocks.columns = df_stocks.columns.str.strip()

# Keep only columns we need
df_stocks = df_stocks[["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "company"]]

# Remove timezone from date
#df_stocks["Date"] = pd.to_datetime(df_stocks["Date"]).dt.date
df_stocks["Date"] = pd.to_datetime(df_stocks["Date"], utc=True).dt.date

# Save clean stock data
df_stocks.to_csv("data/processed/stocks_clean.csv", index=False)

print("Stock data after cleaning:")
print("Shape:", df_stocks.shape)
print("Clean stock data saved to data/processed/stocks_clean.csv")