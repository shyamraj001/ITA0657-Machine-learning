import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Dataset (non-linear pattern)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])  # y = x^2

# -------- Linear Regression --------
lin_model = LinearRegression()
lin_model.fit(X, y)
y_lin_pred = lin_model.predict(X)

# -------- Polynomial Regression --------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# -------- Evaluation --------
print("Linear Regression:")
print("MSE:", mean_squared_error(y, y_lin_pred))
print("R2:", r2_score(y, y_lin_pred))

print("\nPolynomial Regression:")
print("MSE:", mean_squared_error(y, y_poly_pred))
print("R2:", r2_score(y, y_poly_pred))

# -------- Visualization --------
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_lin_pred, label="Linear Regression")
plt.plot(X, y_poly_pred, label="Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()