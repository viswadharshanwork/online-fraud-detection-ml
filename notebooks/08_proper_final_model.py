import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ==========================================
# PROPER FINAL MODEL PIPELINE
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

# Remove exact duplicates
df = df.drop_duplicates()

print("Dataset shape:", df.shape)


# ==========================================
# 1. FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# 2. FIRST SPLIT
# ==========================================
# 80% temporary training data
# 20% final test data

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. SECOND SPLIT
# ==========================================
# From the 80% temporary data:
# 75% -> training
# 25% -> validation
#
# Final proportions:
# 60% training
# 20% validation
# 20% testing

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


print("\n========== DATA SPLIT ==========")

print("Training   :", X_train.shape)
print("Validation :", X_validation.shape)
print("Testing    :", X_test.shape)


# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 5. VALIDATION PREDICTIONS
# ==========================================

validation_probability = model.predict_proba(
    X_validation
)[:, 1]


# ==========================================
# 6. DEFAULT THRESHOLD
# ==========================================

validation_prediction = (
    validation_probability >= 0.5
).astype(int)


print("\n========== VALIDATION RESULTS ==========")

print(
    classification_report(
        y_validation,
        validation_prediction,
        digits=4
    )
)


# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

test_probability = model.predict_proba(
    X_test
)[:, 1]

test_prediction = (
    test_probability >= 0.5
).astype(int)


print("\n========== FINAL TEST CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        test_prediction
    )
)


print("\n========== FINAL TEST CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        test_prediction,
        digits=4
    )
)


# ==========================================
# 8. ROC-AUC
# ==========================================

roc_auc = roc_auc_score(
    y_test,
    test_probability
)

print("\n========== FINAL TEST ROC-AUC ==========")
print(f"{roc_auc:.4f}")


# ==========================================
# 9. PR-AUC
# ==========================================

pr_auc = average_precision_score(
    y_test,
    test_probability
)

print("\n========== FINAL TEST PR-AUC ==========")
print(f"{pr_auc:.4f}")


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/final_random_forest.pkl"
)


# Save threshold
threshold = 0.5

joblib.dump(
    threshold,
    "models/fraud_threshold.pkl"
)


print("\n========== SAVING ==========")
print("Model saved:")
print("models/final_random_forest.pkl")

print("Threshold saved:")
print("models/fraud_threshold.pkl")

print("\nFINAL MODEL PIPELINE COMPLETED.")