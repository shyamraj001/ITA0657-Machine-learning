import numpy as np
from scipy.stats import multivariate_normal

# -----------------------------
# Generate Sample Data
# -----------------------------
np.random.seed(42)

# Two clusters
mean1 = [2, 2]
cov1 = [[1, 0.5], [0.5, 1]]

mean2 = [7, 7]
cov2 = [[1, -0.3], [-0.3, 1]]

data1 = np.random.multivariate_normal(mean1, cov1, 150)
data2 = np.random.multivariate_normal(mean2, cov2, 150)

X = np.vstack((data1, data2))
n_samples, n_features = X.shape

# -----------------------------
# Initialize Parameters
# -----------------------------
k = 2  # number of clusters

np.random.seed(0)
means = X[np.random.choice(n_samples, k, replace=False)]
covariances = [np.eye(n_features) for _ in range(k)]
weights = np.ones(k) / k

# -----------------------------
# EM Algorithm
# -----------------------------
def expectation(X, means, covariances, weights):
    responsibilities = np.zeros((n_samples, k))
    
    for i in range(k):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        responsibilities[:, i] = weights[i] * rv.pdf(X)
    
    # Normalize
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def maximization(X, responsibilities):
    Nk = responsibilities.sum(axis=0)
    
    means_new = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
    
    covariances_new = []
    for i in range(k):
        diff = X - means_new[i]
        cov = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]
        covariances_new.append(cov)
    
    weights_new = Nk / n_samples
    
    return means_new, covariances_new, weights_new

# -----------------------------
# Run EM Iterations
# -----------------------------
iterations = 20

for _ in range(iterations):
    responsibilities = expectation(X, means, covariances, weights)
    means, covariances, weights = maximization(X, responsibilities)

# -----------------------------
# Output Results
# -----------------------------
print("Final Means:\n", means)
print("\nFinal Weights:\n", weights)
print("\nFinal Covariances:\n", covariances)
