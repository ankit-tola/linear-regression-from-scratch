import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionScratch

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train with Gradient Descent
model_gd = LinearRegressionScratch(method="gd", alpha=0.1, epochs=1000)
model_gd.fit(X, y)
y_pred_gd = model_gd.predict(X)

# Train with Normal Equation
model_ne = LinearRegressionScratch(method="normal")
model_ne.fit(X, y)
y_pred_ne = model_ne.predict(X)

# Plot results
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred_gd, "r--", label="Gradient Descent")
plt.plot(X, y_pred_ne, "g-", label="Normal Equation")
plt.legend()
plt.show()

print("GD coefficients:", model_gd.theta)
print("Normal Eq coefficients:", model_ne.theta)
