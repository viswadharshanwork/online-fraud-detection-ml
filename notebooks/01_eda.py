import pandas as pd
import numpy as np

# ==========================================
# ONLINE TRANSACTION FRAUD PREDICTION
# Exploratory Data Analysis
# ==========================================

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== CLASS DISTRIBUTION ==========")
print(df["Class"].value_counts())

print("\n========== CLASS PERCENTAGE ==========")
print(df["Class"].value_counts(normalize=True) * 100)

print("\n========== BASIC STATISTICS ==========")
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# VISUALIZATION
# ==========================================

# 1. Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()


# 2. Transaction Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()


# 3. Transaction Time Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Time"], bins=50)
plt.title("Transaction Time Distribution")
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Transactions")
plt.show()


# 4. Amount by Class
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.show()

# ==========================================
# DEEPER EDA
# ==========================================

# 5. Fraud vs Legitimate - Transaction Amount
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=100,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")
plt.xlim(0, 1000)
plt.show()


# 6. Log-scaled Transaction Amount
plt.figure(figsize=(8, 5))

sns.histplot(
    np.log1p(df["Amount"]),
    bins=100
)

plt.title("Log-Transformed Transaction Amount")
plt.xlabel("log(1 + Amount)")
plt.ylabel("Frequency")
plt.show()


# 7. Duplicate Rows by Class
duplicates = df[df.duplicated(keep=False)]

print("\n========== DUPLICATE ROWS BY CLASS ==========")
print(duplicates["Class"].value_counts())

print("\n========== FRAUD DUPLICATE ROWS ==========")
print(duplicates[duplicates["Class"] == 1].shape[0])

print("\n========== LEGITIMATE DUPLICATE ROWS ==========")
print(duplicates[duplicates["Class"] == 0].shape[0])

# ==========================================
# DUPLICATE CONSISTENCY CHECK
# ==========================================

# Features excluding the target column
feature_columns = df.columns.drop("Class")

# Find duplicate feature combinations
duplicate_features = df[
    df.duplicated(subset=feature_columns, keep=False)
].copy()

print("\n========== DUPLICATE FEATURE ROWS ==========")
print("Rows involved:", len(duplicate_features))

# Check whether identical feature rows have different Class labels
conflicting_duplicates = (
    duplicate_features
    .groupby(list(feature_columns))["Class"]
    .nunique()
)

conflicting_count = (conflicting_duplicates > 1).sum()

print("\n========== CONFLICTING DUPLICATES ==========")
print("Duplicate groups with different Class labels:", conflicting_count)

# Compare dataset before and after removing exact duplicates
df_without_duplicates = df.drop_duplicates()

print("\n========== AFTER REMOVING DUPLICATES ==========")
print("Original rows:", len(df))
print("Rows after removing duplicates:", len(df_without_duplicates))

print("\n========== CLASS DISTRIBUTION AFTER DUPLICATE REMOVAL ==========")
print(df_without_duplicates["Class"].value_counts())

print("\n========== FRAUD COUNT AFTER DUPLICATE REMOVAL ==========")
print(
    "Fraud transactions:",
    (df_without_duplicates["Class"] == 1).sum()
)