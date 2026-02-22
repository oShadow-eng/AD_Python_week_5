"""
Day 24 Activity: Feature Selection
Tasks:
1) Load dataset
2) Apply variance threshold
3) Remove highly correlated features
4) Compare model performance before/after
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np 
#Task 1 Load dataset
df = pd.read_csv('data_day24_selection.csv')

#Task 2 Apply variance threshold
variance_threshold = VarianceThreshold(threshold=0.0)
x_selected = variance_threshold.fit_transform(df)

#Task 3 Remove highly correlated features
df_selected = pd.DataFrame(x_selected, columns=df.columns[variance_threshold.get_support()])
#Task 4 Compare model performance before/after
print(f"Before feature selection: {df}")
print(f"after feature selection: {df_selected}")




# TODO: Apply variance threshold
# TODO: Drop correlated features
