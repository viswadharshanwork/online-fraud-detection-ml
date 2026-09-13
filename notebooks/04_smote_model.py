import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# ==========================================
# SMOTE + LOGISTIC REGRESSION
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


# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ==========================================
# 5. SMOTE
# ==========================================

print("\n========== BEFORE SMOTE ==========")
print(y_train.value_counts())

smote = SMOTE(
    sampling_strategy=0.1,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\n========== AFTER SMOTE ==========")
print(y_train_smote.value_counts())

print("\nTraining data before SMOTE:", X_train.shape)
print("Training data after SMOTE :", X_train_smote.shape)


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

print("\nLogistic Regression training completed.")


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ==========================================
# 8. EVALUATION
# ==========================================

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))


print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))


print("\n========== ROC-AUC ==========")
print(roc_auc_score(y_test, y_probability))