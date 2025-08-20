import numpy as np

class LinearRegressionScratch:
    def __init__(self, method="gd", alpha=0.01, epochs=1000):
        self.method = method
        self.alpha = alpha
        self.epochs = epochs
        self.theta = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _normalize(self, X, y):
        """Standardize features and target for stability (Gradient Descent only)."""
        self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
        self.y_mean, self.y_std = y.mean(), y.std()

        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std
        return X_norm, y_norm

    def fit(self, X, y):
        m = len(y)

        # If gradient descent: normalize data
        if self.method == "gd":
            X, y = self._normalize(X, y)

        # Add bias column
        X_b = np.c_[np.ones((m, 1)), X]

        if self.method == "normal":
            # Normal Equation (no scaling needed)
            self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        else:
            # Gradient Descent
            self.theta = np.random.randn(X_b.shape[1])
            for _ in range(self.epochs):
                gradients = 2/m * X_b.T.dot(X_b.dot(self.theta) - y)
                self.theta -= self.alpha * gradients

    def predict(self, X):
        m = len(X)

        # Normalize X if using GD
        if self.method == "gd":
            X = (X - self.X_mean) / self.X_std

        X_b = np.c_[np.ones((m, 1)), X]
        y_pred = X_b.dot(self.theta)

        # Denormalize if GD
        if self.method == "gd":
            y_pred = y_pred * self.y_std + self.y_mean

        return y_pred
