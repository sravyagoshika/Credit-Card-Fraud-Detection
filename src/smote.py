import pandas as pd
import os

from imblearn.over_sampling import SMOTE

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - SMOTE")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load processed training and testing data
# ------------------------------------------------------------

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
)

# Convert target DataFrames to Series
y_train = y_train["is_fraud"]
y_test = y_test["is_fraud"]

# ------------------------------------------------------------
# 2. Display original class distribution
# ------------------------------------------------------------

print("\nOriginal training class distribution:")

print(y_train.value_counts())

print("\nOriginal training percentages:")

print(
    y_train.value_counts(normalize=True) * 100
)

# ------------------------------------------------------------
# 3. Apply SMOTE
# ------------------------------------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# ------------------------------------------------------------
# 4. Display new class distribution
# ------------------------------------------------------------

print("\nClass distribution after SMOTE:")

print(
    y_train_smote.value_counts()
)

print("\nPercentages after SMOTE:")

print(
    y_train_smote.value_counts(normalize=True) * 100
)

# ------------------------------------------------------------
# 5. Display shapes
# ------------------------------------------------------------

print("\nBefore SMOTE:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nAfter SMOTE:")
print("X_train_smote:", X_train_smote.shape)
print("y_train_smote:", y_train_smote.shape)

print("\nTest data remains unchanged:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ------------------------------------------------------------
# 6. Save SMOTE datasets
# ------------------------------------------------------------

os.makedirs(
    "data/processed/smote",
    exist_ok=True
)

X_train_smote.to_csv(
    "data/processed/smote/X_train_smote.csv",
    index=False
)

y_train_smote.to_csv(
    "data/processed/smote/y_train_smote.csv",
    index=False
)

# ------------------------------------------------------------
# 7. Save original test data again
# ------------------------------------------------------------

X_test.to_csv(
    "data/processed/smote/X_test.csv",
    index=False
)

y_test.to_csv(
    "data/processed/smote/y_test.csv",
    index=False
)

print("\nSMOTE processing completed successfully!")

print("\nFiles saved in:")
print("data/processed/smote/")

print("=" * 60)