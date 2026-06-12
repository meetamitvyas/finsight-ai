import pandas as pd

df = pd.read_csv("data/processed/credit_clean.csv")

print("RevolvingUtilizationOfUnsecuredLines statistics:")
print(df["RevolvingUtilizationOfUnsecuredLines"].describe())
print()
print("Top 10 highest values:")
print(df["RevolvingUtilizationOfUnsecuredLines"].sort_values(ascending=False).head(10))
print()
print("How many values are above 1.0?")
print((df["RevolvingUtilizationOfUnsecuredLines"] > 1).sum())
print()
print("How many values are above 10?")
print((df["RevolvingUtilizationOfUnsecuredLines"] > 10).sum())