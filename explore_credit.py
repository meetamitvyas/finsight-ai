import pandas as pd

# Load the credit risk training data
df = pd.read_csv("data/raw/credit/cs-training.csv")

# Basic information
print("Shape of data (rows, columns):", df.shape)
print()
print("Column names:")
print(df.columns.tolist())
print()
print("First 3 rows:")
print(df.head(3))
print()
print("How many people defaulted vs did not default:")
print(df["SeriousDlqin2yrs"].value_counts())
print()
print("Missing values in each column:")
print(df.isnull().sum())