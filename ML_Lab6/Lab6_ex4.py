from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing


[X,y]=fetch_california_housing(return_X_y=True)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=369)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

degrees = [1, 2, 3, 4, 5]
best_degree = 0
best_r2_val = 0
# best_r2_val = -float("inf")

for degree in degrees:
    print(f"Training Polynomial Regression model with degree {degree}")

    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train_scaled)
    X_poly_val = poly.transform(X_val_scaled)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_val_predictions= model.predict(X_poly_val)
    r2_val = r2_score(y_val, y_val_predictions)

    print(f" Validation R2: {r2_val}")
    print(f"Degree: {degree}, Validation R²: {r2_val:.4f}")
    print()

    if r2_val > best_r2_val:
        best_r2_val = r2_val
        best_degree = degree
        best_model = model

print(f"Best Polynomial Degree: {best_degree}")

poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

final_model = LinearRegression()
final_model.fit(X_train_poly, y_train)

y_test_pred = final_model.predict(X_test_poly)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test R² Score: {r2_test:.4f}")

#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # Check distribution of features
# plt.figure(figsize=(10, 6))
# sns.histplot(X_train_scaled[:, 0], kde=True)  # For first feature
# plt.title("Feature Distribution")
# plt.show()
#
# # Check for outliers (boxplot for first feature)
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=X_train_scaled[:, 0])  # For first feature
# plt.title("Boxplot of Feature")
# plt.show()
#
# # Check residuals for model with degree 1
# y_train_pred = best_model.predict(poly.fit_transform(X_train_scaled))
# residuals = y_train - y_train_pred
#
# plt.figure(figsize=(10, 6))
# plt.scatter(y_train_pred, residuals)
# plt.hlines(0, min(y_train_pred), max(y_train_pred), colors='red')
# plt.xlabel("Predicted values")
# plt.ylabel("Residuals")
# plt.title("Residuals vs Predicted Values")
# plt.show()




