import sqlite3
import pandas as pd

# Connect to our warehouse
conn = sqlite3.connect("finsight.db")

print("=" * 50)
print("QUERY 1: How many people defaulted vs did not?")
print("=" * 50)
result = pd.read_sql("""
    SELECT 
        SeriousDlqin2yrs as defaulted,
        COUNT(*) as total_people,
        ROUND(COUNT(*) * 100.0 / 150000, 2) as percentage
    FROM credit_risk
    GROUP BY SeriousDlqin2yrs
""", conn)
print(result)

print()
print("=" * 50)
print("QUERY 2: Average debt ratio by age group")
print("=" * 50)
result2 = pd.read_sql("""
    SELECT 
        CASE 
            WHEN age < 30 THEN 'Under 30'
            WHEN age BETWEEN 30 AND 50 THEN '30 to 50'
            ELSE 'Over 50'
        END as age_group,
        ROUND(AVG(DebtRatio), 2) as avg_debt_ratio,
        ROUND(AVG(MonthlyIncome), 0) as avg_monthly_income,
        COUNT(*) as total_people
    FROM credit_risk
    GROUP BY age_group
""", conn)
print(result2)

print()
print("=" * 50)
print("QUERY 3: Latest stock price for each company")
print("=" * 50)
result3 = pd.read_sql("""
    SELECT 
        company,
        ticker,
        Date,
        ROUND(Close, 2) as latest_price
    FROM stock_prices
    WHERE Date = (SELECT MAX(Date) FROM stock_prices)
    ORDER BY latest_price DESC
""", conn)
print(result3)

conn.close()