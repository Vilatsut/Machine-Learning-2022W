# %%
import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights, self.bias = None, None
    
    def fit(self, X, y):
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # 2. Perform gradient descent calculation
        for i in range(self.n_iterations):
            # Line equation
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate derivatives (of MSE)
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_pred - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_pred - y))
            
            # Update the coefficients
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
# %%
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = data.data
y = data.target

# %%
