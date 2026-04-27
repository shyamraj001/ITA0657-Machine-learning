# -----------------------------------
# Iris Flower Classification using KNN
# -----------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Load Dataset
# -----------------------------------
iris = load_iris()
X = iris.data       # Features
y = iris.target     # Labels

# -----------------------------------
# Split Data
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Create KNN Model
# -----------------------------------
k = 3
model = KNeighborsClassifier(n_neighbors=k)

# -----------------------------------
# Train Model
# -----------------------------------
model.fit(X_train, y_train)

# -----------------------------------
# Prediction
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation
# -----------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# Test with New Sample
# -----------------------------------
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("\nPredicted Class:", iris.target_names[prediction][0])
