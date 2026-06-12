import pandas as pd
import numpy as np

# ── Load our original (reference) data ────────────────────────────────
df = pd.read_csv("data/processed/credit_clean.csv")

# ── Simulate "new" data with drift ─────────────────────────────────────
# We take a random sample of 5000 customers to represent "this month's" data
new_data = df.sample(n=5000, random_state=99).copy()

# Simulate economic change: incomes have generally increased (inflation)
# We multiply MonthlyIncome by 1.3 (30% increase)
new_data["MonthlyIncome"] = new_data["MonthlyIncome"] * 1.3

# Simulate behavior change: people are using more of their credit limit
# We increase RevolvingUtilizationOfUnsecuredLines by 20%, capped at reasonable values
new_data["RevolvingUtilizationOfUnsecuredLines"] = (
    new_data["RevolvingUtilizationOfUnsecuredLines"] * 1.2
).clip(upper=2.0)

# Save this as our "current" data
new_data.to_csv("data/processed/credit_current.csv", index=False)

print("Simulated 'current' data created with drift:")
print(f"  Rows: {len(new_data)}")
print()
print("Average MonthlyIncome:")
print(f"  Reference (original): {df['MonthlyIncome'].mean():.0f}")
print(f"  Current (simulated):  {new_data['MonthlyIncome'].mean():.0f}")
print()
print("Average RevolvingUtilizationOfUnsecuredLines:")
print(f"  Reference (original): {df['RevolvingUtilizationOfUnsecuredLines'].mean():.2f}")
print(f"  Current (simulated):  {new_data['RevolvingUtilizationOfUnsecuredLines'].mean():.2f}")