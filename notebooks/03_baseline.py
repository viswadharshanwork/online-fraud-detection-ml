import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# BASELINE LOGISTIC REGRESSION
# ==========================================

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Remove exact duplicates
df = df.drop_duplicates()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Logistic Regression training completed.")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# EVALUATION
# ==========================================

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

print("\n========== ROC-AUC ==========")
print(roc_auc_score(y_test, y_probability))