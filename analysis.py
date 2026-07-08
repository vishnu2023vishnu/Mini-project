import pandas as pd

data = pd.read_csv("placementdata.csv")

print(data.head())

print("\nShape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())