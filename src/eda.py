import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/financial_fraud_dataset.csv")

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - DATA EXPLORATION")
print("=" * 60)

# Dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Columns:")
print(df.columns.tolist())

# First 5 records
print("\n3. First 5 Records:")
print(df.head())

# Data types
print("\n4. Data Types:")
print(df.dtypes)

# Missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Duplicate records
print("\n6. Duplicate Records:")
print(df.duplicated().sum())

# Fraud distribution
print("\n7. Fraud Distribution:")
print(df["is_fraud"].value_counts())

# Fraud percentage
fraud_percentage = (
    df["is_fraud"].value_counts(normalize=True) * 100
)

print("\n8. Fraud Percentage:")
print(fraud_percentage)

# Statistical summary
print("\n9. Statistical Summary:")
print(df.describe())

# Merchant category distribution
print("\n10. Merchant Categories:")
print(df["merchant_category"].value_counts())

# Device type distribution
print("\n11. Device Types:")
print(df["device_type"].value_counts())

# Customer location distribution
print("\n12. Customer Locations:")
print(df["customer_location"].value_counts())

# ----------------------------------------
# VISUALIZATIONS
# ----------------------------------------

# Fraud distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="is_fraud"
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.savefig(
    "reports/figures/fraud_distribution.png"
)

plt.close()


# Transaction amount distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="amount",
    bins=50,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.savefig(
    "reports/figures/amount_distribution.png"
)

plt.close()


# Fraud by merchant category
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="merchant_category",
    hue="is_fraud"
)

plt.title("Fraud Transactions by Merchant Category")
plt.xlabel("Merchant Category")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=30)

plt.savefig(
    "reports/figures/fraud_by_merchant.png"
)

plt.close()


# Fraud by device type
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="device_type",
    hue="is_fraud"
)

plt.title("Fraud Transactions by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Number of Transactions")

plt.savefig(
    "reports/figures/fraud_by_device.png"
)

plt.close()


print("\nEDA completed successfully!")