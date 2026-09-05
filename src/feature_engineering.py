import pandas as pd
import numpy as np
import os

# ============================================================
# CREDIT CARD FRAUD DETECTION
# FEATURE ENGINEERING
# ============================================================

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

input_file = "data/financial_fraud_dataset.csv"

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. Convert timestamp to datetime
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by customer and time
df = df.sort_values(
    by=["customer_id", "timestamp"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 3. Extract time-based features
# ------------------------------------------------------------

df["transaction_hour"] = df["timestamp"].dt.hour

df["transaction_day"] = df["timestamp"].dt.day

df["transaction_month"] = df["timestamp"].dt.month

df["transaction_day_of_week"] = df["timestamp"].dt.dayofweek

# Saturday = 5, Sunday = 6
df["is_weekend"] = (
    df["transaction_day_of_week"] >= 5
).astype(int)

# Night transactions: 10 PM - 5 AM
df["is_night"] = (
    (df["transaction_hour"] >= 22) |
    (df["transaction_hour"] <= 5)
).astype(int)

# ------------------------------------------------------------
# 4. Customer historical average transaction amount
# ------------------------------------------------------------

# Shift by one transaction so the current transaction
# does not influence its own historical average.

df["customer_avg_amount"] = (
    df.groupby("customer_id")["amount"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

# For the customer's first transaction, use overall average
overall_avg = df["amount"].mean()

df["customer_avg_amount"] = (
    df["customer_avg_amount"]
    .fillna(overall_avg)
)

# ------------------------------------------------------------
# 5. Amount deviation
# ------------------------------------------------------------

df["amount_deviation"] = (
    df["amount"] - df["customer_avg_amount"]
)

# ------------------------------------------------------------
# 6. Amount ratio
# ------------------------------------------------------------

df["amount_ratio"] = (
    df["amount"] /
    (df["customer_avg_amount"] + 1e-6)
)

# ------------------------------------------------------------
# 7. Clean numerical values
# ------------------------------------------------------------

df["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce"
)

df["customer_age"] = pd.to_numeric(
    df["customer_age"],
    errors="coerce"
)

df["previous_transactions"] = pd.to_numeric(
    df["previous_transactions"],
    errors="coerce"
)

# Remove rows with invalid numerical values
df = df.dropna(
    subset=[
        "amount",
        "customer_age",
        "previous_transactions"
    ]
)

# ------------------------------------------------------------
# 8. Remove unnecessary identifier column
# ------------------------------------------------------------

# transaction_id and customer_id are identifiers.
# They should not be directly given to the ML model.

model_df = df.drop(
    columns=[
        "transaction_id",
        "customer_id",
        "timestamp"
    ]
)

# ------------------------------------------------------------
# 9. Save engineered dataset
# ------------------------------------------------------------

output_file = "data/engineered_fraud_dataset.csv"

model_df.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# 10. Display results
# ------------------------------------------------------------

print("\nFeature engineering completed successfully!")

print("\nNew dataset shape:")
print(model_df.shape)

print("\nNew features created:")

new_features = [
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

for feature in new_features:
    print(" -", feature)

print("\nFinal columns:")
print(model_df.columns.tolist())

print("\nFirst 5 rows:")
print(model_df.head())

print("\nSaved engineered dataset to:")
print(output_file)

print("=" * 60)