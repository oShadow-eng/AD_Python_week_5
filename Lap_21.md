"""
Day 21 Activity: Domain-Driven Features
Tasks:
1) Load housing data
2) Create price_per_sqft safely
3) Propose at least two additional domain features
4) Compare basic model behavior (or summaries) with/without domain features
"""

import pandas as pd
import numpy as np
#Task 1: Load housing data
df = pd.read_csv("day21_housing.csv")

#Task 2: Create price_per_sqft safely
# To avoid division by zero, we can use np.where to create the price_per_sqft column
safe_sqft = df["sqft"].replace(0, np.nan)

df["price_per_sqft"] = df["price"] / safe_sqft

df[["price", "sqft", "price_per_sqft"]].head()

#Task 3: Create additional domain features
# Example domain features:
# Total rooms (bedrooms + bathrooms)
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

#bathroom for evrey bedroom
df["bathroom_per_bedroom"] = np.where(df["bedrooms"] > 0, df["bathrooms"] / df["bedrooms"], np.nan)


#Task 4: Compare basic model behavior (or summaries) with/without domain features
# For simplicity, we can compare the correlation of features with price
# Correlation without domain features
print("Correlation without domain features:")
print(df[["price", "sqft", "bedrooms", "bathrooms"]].corr()["price"])
# Correlation with domain features
print("\nCorrelation with all features:")
print(df[["price", "sqft", "bedrooms", "bathrooms", "price_per_sqft", "total_rooms", "bathroom_per_bedroom"]].corr()["price"])
# TODO: Create price_per_sqft with safe division
# TODO: Create additional domain features
# TODO: Print summary or model comparison
