"""
Day 22 Activity: Interaction Features
Tasks:
1) Load interaction dataset
2) Create multiplicative, additive, and logical interactions
3) Compute correlations with target
"""

import pandas as pd

#Task 1: Load interaction dataset
df = pd.read_csv("day22_interactions.csv")

#Task 2: Create interaction features
# Multiplicative interaction between feature1 and feature2
df['feature1_feature2_mult'] = df['feature1'] * df['feature2']
# Additive interaction between feature1 and feature2
df['feature1_feature2_add'] = df['feature1'] + df['feature2']
# Logical interaction (AND) between feature1 and feature2
df['feature1_feature2_and'] = (df['feature1'] > 0) & (df['feature2'] > 0)

#Task 3: Compute correlations with target
correlations = df.corr()['target'].sort_values(ascending=False)
print(correlations)
# TODO: Create interaction features
# TODO: Compute correlations with target
