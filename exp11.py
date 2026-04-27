# -----------------------------------
# Credit Score Classification Program (Fixed)
# -----------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Load Dataset
# -----------------------------------
data = pd.read_csv("credit_score.csv")

# -----------------------------------
# Handle Missing Values
# -----------------------------------
data = data.dropna()

# -----------------------------------
# Separate Features and Target FIRST
# -----------------------------------
X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]

# -----------------------------------
# Encode Target Variable
# -----------------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# -----------------------------------
# Encode Categorical Features ONLY
# -----------------------------------
label_encoders = {}
for col in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -----------------------------------
# Split Data
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Feature Scaling (Optional for RF, but OK)
# -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# Model Training
# -----------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
# Decode Predictions (Optional)
# -----------------------------------
decoded_preds = target_encoder.inverse_transform(y_pred)
print("\nSample Predictions (Decoded):", decoded_preds[:5])
