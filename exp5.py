import numpy as np
from collections import Counter

# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Class
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    # Training function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Prediction function
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    # Helper function for a single prediction
    def _predict(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Sample dataset
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 6]
    ])
    
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Test data
    X_test = np.array([
        [2, 2],
        [7, 6]
    ])
    
    # Model
    model = KNN(k=3)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("Predictions:", predictions)