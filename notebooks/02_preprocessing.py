import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# ONLINE TRANSACTION FRAUD PREDICTION
# DATA PREPROCESSING
# ==========================================

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Original dataset shape:", df.shape)

# ==========================================
# 1. REMOVE EXACT DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)

# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

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

print("\n========== TRAIN-TEST SPLIT ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ==========================================
# 4. CLASS DISTRIBUTION
# ==========================================

print("\n========== TRAINING CLASS DISTRIBUTION ==========")
print(y_train.value_counts())

print("\n========== TEST CLASS DISTRIBUTION ==========")
print(y_test.value_counts())

print("\n========== TRAINING CLASS PERCENTAGE ==========")
print(y_train.value_counts(normalize=True) * 100)

print("\n========== TEST CLASS PERCENTAGE ==========")
print(y_test.value_counts(normalize=True) * 100)