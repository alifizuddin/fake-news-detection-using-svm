import pandas as pd

# Load the data
df = pd.read_csv("dataset/FakeNewsNet.csv")

# View first 5 rows
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nLabel counts:\n", df["real"].value_counts())