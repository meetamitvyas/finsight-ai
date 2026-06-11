import pandas as pd

# Load our clean credit data
df = pd.read_csv("data/processed/credit_clean.csv")

# Separate features (X) from target (y)
# Features = everything the model learns FROM
# Target = what the model is trying to PREDICT
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print()
print("Features we are using:")
print(X.columns.tolist())
print()
print("Target distribution:")
print(y.value_counts())