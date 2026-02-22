"""""
Day 25 Activity: Mini-Project (Feature Engineering)
Tasks:
1) Load dataset
2) Apply domain features, interactions, and transformations
3) Apply feature selection
4) Save engineered dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
#Task 1: Load dataset
df = pd.read_csv('day25_project.csv')

#Task 2: Apply domain features, interactions, and transformations
# Apply domain features
price = df['price']
sqft = df['sqft']
room = df['rooms']
price_per_sqft = price / sqft
room_per_sqft = room / sqft

# Apply interactions
df['price_per_sqft'] = price_per_sqft
df['room_per_sqft'] = room_per_sqft

#transformations
price_power_room = price * room
price_slash_sqft = price / sqft
room_power_sqft = room * sqft

df['price_power_room'] = price_power_room
df['price_power_sqft'] = price_slash_sqft
df['room_power_sqft'] = room_power_sqft

#Task 3: Apply feature selection

# Apply variance threshold
variance_threshold = VarianceThreshold(threshold=0.0)
x_selected = variance_threshold.fit_transform(df)

# رجّع الـ array إلى DataFrame مع الأعمدة اللي تم الاحتفاظ بها
df_selected = pd.DataFrame(x_selected, columns=df.columns[variance_threshold.get_support()])

#Task 4: Save engineered dataset
df_selected.to_csv('day25_engineered.csv', index=False)


# TODO: Build engineered feature set
# TODO: Save engineered dataset to data/day25_engineered.csv
