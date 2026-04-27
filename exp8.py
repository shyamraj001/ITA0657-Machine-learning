import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 4, 5, 4, 5, 6, 7, 8, 9, 10])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training Data (X_train):\n", X_train)
print("Testing Data (X_test):\n", X_test)

print("\nActual Values (y_test):", y_test)
print("Predicted Values:", y_pred)

print("\nMean Squared Error (MSE):", mse)
print("R2 Score:", r2)

new_input = np.array([[11]])
new_prediction = model.predict(new_input)
print("\nPrediction for input 11:", new_prediction)