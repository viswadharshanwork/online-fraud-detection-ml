import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================
# RANDOM FOREST FRAUD DETECTION
# ==========================================

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# ==========================================
# 1. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ==========================================
# 4. RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed.")


# ==========================================
# 5. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 6. EVALUATION
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))


print("\n========== ROC-AUC ==========")

print(roc_auc_score(y_test, y_probability))