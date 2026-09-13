import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt


# ==========================================
# FINAL RANDOM FOREST MODEL
# ==========================================

# Load dataset
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
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. FINAL RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining final Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 4. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== FINAL CONFUSION MATRIX ==========")
print(cm)


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\n========== FINAL CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ==========================================
# 7. ROC-AUC
# ==========================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== ROC-AUC ==========")
print(f"{roc_auc:.4f}")


# ==========================================
# 8. PRECISION-RECALL AUC
# ==========================================

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\n========== PRECISION-RECALL AUC ==========")
print(f"{pr_auc:.4f}")


# ==========================================
# 9. PRECISION-RECALL CURVE
# ==========================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 5))

plt.plot(
    recall,
    precision
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.grid(True)

plt.show()


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")

print(
    feature_importance.head(10)
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/final_random_forest.pkl"
)

print("\nFinal model saved successfully.")