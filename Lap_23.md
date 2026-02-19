"""
Day 23 Activity: Polynomial Features
Tasks:
1) Load regression dataset
2) Add polynomial features (degrees 1,2,5)
3) Compare model fits or visualize predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# TODO: Load data from data/day23_poly.csv
df = pd.read_csv("day23_poly.csv")

#Task 2: Add polynomial features (degrees 1,2,5)
X = df[["x"]].values
y = df["y"].values
# Create polynomial features
poly_1 = PolynomialFeatures(degree=1)
poly_2 = PolynomialFeatures(degree=2)
poly_5 = PolynomialFeatures(degree=5)
X_poly_1 = poly_1.fit_transform(X)
X_poly_2 = poly_2.fit_transform(X)
X_poly_5 = poly_5.fit_transform(X)
# Fit linear regression models
model_1 = LinearRegression().fit(X_poly_1, y)
model_2 = LinearRegression().fit(X_poly_2, y)
model_5 = LinearRegression().fit(X_poly_5, y)
# Task 3: Compare model fits or visualize predictions

#Printing coefficients for each model
print('Before adding polynomial features:', df)
print("Coefficients for Degree 1:", model_1.coef_)
print("Coefficients for Degree 2:", model_2.coef_)
print("Coefficients for Degree 5:", model_5.coef_)

# Generate predictions for plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly_1 = poly_1.transform(X_plot)
X_plot_poly_2 = poly_2.transform(X_plot)
X_plot_poly_5 = poly_5.transform(X_plot)
y_plot_1 = model_1.predict(X_plot_poly_1)
y_plot_2 = model_2.predict(X_plot_poly_2)
y_plot_5 = model_5.predict(X_plot_poly_5)
# Plotting
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_plot, y_plot_1, color='red', label='Degree 1')
plt.plot(X_plot, y_plot_2, color='green', label='Degree 2')
plt.plot(X_plot, y_plot_5, color='orange', label='Degree 5')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Fits')
plt.legend()
plt.show()
