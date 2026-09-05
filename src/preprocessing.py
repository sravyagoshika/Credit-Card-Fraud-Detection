import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - DATA PREPROCESSING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load engineered dataset
# ------------------------------------------------------------

input_file = "data/engineered_fraud_dataset.csv"

df = pd.read_csv(input_file)

print("\nDataset shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# 3. Identify categorical and numerical columns
# ------------------------------------------------------------

categorical_columns = [
    "merchant_category",
    "customer_location",
    "device_type"
]

numerical_columns = [
    "amount",
    "customer_age",
    "previous_transactions",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_day_of_week",
    "is_weekend",
    "is_night",
    "customer_avg_amount",
    "amount_deviation",
    "amount_ratio"
]

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# ------------------------------------------------------------
# 4. One-hot encode categorical features
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nShape after encoding:")
print(X.shape)

# ------------------------------------------------------------
# 5. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

# ------------------------------------------------------------
# 6. Scale numerical features
# ------------------------------------------------------------

scaler = StandardScaler()

# Only scale columns that exist after encoding
scale_columns = [
    column
    for column in numerical_columns
    if column in X_train.columns
]

X_train[scale_columns] = scaler.fit_transform(
    X_train[scale_columns]
)

X_test[scale_columns] = scaler.transform(
    X_test[scale_columns]
)

print("\nScaled numerical columns:")
print(scale_columns)

# ------------------------------------------------------------
# 7. Create required directories
# ------------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)

# ------------------------------------------------------------
# 8. Save processed datasets
# ------------------------------------------------------------

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

# ------------------------------------------------------------
# 9. Save feature names
# ------------------------------------------------------------

with open(
    "data/processed/feature_names.txt",
    "w"
) as file:

    for column in X_train.columns:
        file.write(column + "\n")

# ------------------------------------------------------------
# 10. Save scaler
# ------------------------------------------------------------

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nScaler saved successfully:")
print("models/scaler.pkl")

# ------------------------------------------------------------
# 11. Save columns that need scaling
# ------------------------------------------------------------

with open(
    "models/scale_columns.txt",
    "w"
) as file:

    for column in scale_columns:
        file.write(column + "\n")

print("\nScale columns saved:")
print("models/scale_columns.txt")

# ------------------------------------------------------------
# 12. Final output
# ------------------------------------------------------------

print("\nPreprocessing completed successfully!")

print("\nFinal training shape:")
print(X_train.shape)

print("\nFinal testing shape:")
print(X_test.shape)

print("\nProcessed files saved in:")
print("data/processed/")

print("\nModel preprocessing files saved:")
print("models/scaler.pkl")
print("models/scale_columns.txt")

print("=" * 60)