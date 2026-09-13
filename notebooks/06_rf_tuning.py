import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# RANDOM FOREST TUNING
# ==========================================

df = pd.read_csv("data/creditcard.csv")

# Remove exact duplicates
df = df.drop_duplicates()

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# MODELS TO COMPARE
# ==========================================

models = {

    "RF_100_balanced": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RF_200_balanced": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RF_100_balanced_subsample": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "RF_200_balanced_subsample": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })


# ==========================================
# RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== RANDOM FOREST TUNING RESULTS ==========")
print(results_df.to_string(index=False))

print("\n========== BEST MODEL BY F1 ==========")

best_model = results_df.loc[
    results_df["F1"].idxmax()
]

print(best_model)