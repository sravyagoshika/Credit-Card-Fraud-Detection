from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib


# ============================================================
# CREDIT CARD FRAUD DETECTION API
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/decision_tree.pkl"

SCALER_PATH = "models/scaler.pkl"

FEATURE_FILE = "data/processed/feature_names.txt"

SCALE_COLUMNS_FILE = "models/scale_columns.txt"

ORIGINAL_DATASET = "data/financial_fraud_dataset.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("Decision Tree model loaded successfully!")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully!")


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

with open(FEATURE_FILE, "r") as file:

    feature_names = [
        line.strip()
        for line in file.readlines()
    ]

print("Feature names loaded successfully!")

print("Number of features:", len(feature_names))


# ============================================================
# LOAD NUMERICAL COLUMNS THAT WERE SCALED
# ============================================================

with open(SCALE_COLUMNS_FILE, "r") as file:

    scale_columns = [
        line.strip()
        for line in file.readlines()
    ]

print("Scale columns loaded successfully!")


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

historical_data = pd.read_csv(
    ORIGINAL_DATASET
)

historical_data["timestamp"] = pd.to_datetime(
    historical_data["timestamp"]
)

print("Historical transaction data loaded successfully!")


# ============================================================
# INPUT DATA MODEL
# ============================================================

class Transaction(BaseModel):

    transaction_id: str

    timestamp: str

    amount: float

    merchant_category: str

    customer_id: str

    customer_age: int

    customer_location: str

    device_type: str

    previous_transactions: int


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API is running",
        "status": "success",
        "model": "Decision Tree"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Decision Tree",
        "number_of_features": len(feature_names),
        "threshold": 0.05
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # --------------------------------------------------------
    # Convert request to dictionary
    # --------------------------------------------------------

    data = transaction.model_dump()

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    timestamp = pd.to_datetime(
        data["timestamp"]
    )

    # --------------------------------------------------------
    # Time-based feature engineering
    # Same logic as feature_engineering.py
    # --------------------------------------------------------

    transaction_hour = timestamp.hour

    transaction_day = timestamp.day

    transaction_month = timestamp.month

    transaction_day_of_week = timestamp.dayofweek

    is_weekend = int(
        transaction_day_of_week >= 5
    )

    is_night = int(
        (transaction_hour >= 22)
        or
        (transaction_hour <= 5)
    )

    # --------------------------------------------------------
    # Find customer's historical transactions
    # --------------------------------------------------------

    customer_history = historical_data[
        (
            historical_data["customer_id"]
            ==
            data["customer_id"]
        )
        &
        (
            historical_data["timestamp"]
            <
            timestamp
        )
    ]

    # --------------------------------------------------------
    # Customer historical average
    # --------------------------------------------------------

    if len(customer_history) > 0:

        customer_avg_amount = (
            customer_history["amount"]
            .mean()
        )

    else:

        customer_avg_amount = (
            historical_data["amount"].mean()
        )

    # --------------------------------------------------------
    # Amount deviation
    # --------------------------------------------------------

    amount_deviation = (
        data["amount"]
        -
        customer_avg_amount
    )

    # --------------------------------------------------------
    # Amount ratio
    # --------------------------------------------------------

    amount_ratio = (
        data["amount"]
        /
        (customer_avg_amount + 1e-6)
    )

    # --------------------------------------------------------
    # Create numerical features
    # --------------------------------------------------------

    features = {

        "amount":
            data["amount"],

        "customer_age":
            data["customer_age"],

        "previous_transactions":
            data["previous_transactions"],

        "transaction_hour":
            transaction_hour,

        "transaction_day":
            transaction_day,

        "transaction_month":
            transaction_month,

        "transaction_day_of_week":
            transaction_day_of_week,

        "is_weekend":
            is_weekend,

        "is_night":
            is_night,

        "customer_avg_amount":
            customer_avg_amount,

        "amount_deviation":
            amount_deviation,

        "amount_ratio":
            amount_ratio
    }

    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    # Create all categorical columns expected by the model.
    # drop_first=True was used during preprocessing.
    # Therefore the first category is represented by all zeros.

    for feature in feature_names:

        if feature.startswith(
            "merchant_category_"
        ):

            features[feature] = 0

        elif feature.startswith(
            "customer_location_"
        ):

            features[feature] = 0

        elif feature.startswith(
            "device_type_"
        ):

            features[feature] = 0

    # --------------------------------------------------------
    # Merchant category
    # --------------------------------------------------------

    merchant_feature = (
        "merchant_category_"
        +
        data["merchant_category"]
    )

    if merchant_feature in features:

        features[merchant_feature] = 1

    # --------------------------------------------------------
    # Customer location
    # --------------------------------------------------------

    location_feature = (
        "customer_location_"
        +
        data["customer_location"]
    )

    if location_feature in features:

        features[location_feature] = 1

    # --------------------------------------------------------
    # Device type
    # --------------------------------------------------------

    device_feature = (
        "device_type_"
        +
        data["device_type"]
    )

    if device_feature in features:

        features[device_feature] = 1

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    input_df = pd.DataFrame(
        [features]
    )

    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # ========================================================
    # SCALE NUMERICAL FEATURES
    # ========================================================

    for column in scale_columns:

        input_df[column] = pd.to_numeric(
            input_df[column],
            errors="coerce"
        )

    input_df[scale_columns] = scaler.transform(
        input_df[scale_columns]
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    probability = model.predict_proba(
        input_df
    )[0][1]

    # ========================================================
    # FRAUD THRESHOLD
    # ========================================================

    threshold = 0.05

    prediction = int(
        probability >= threshold
    )

    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        result = "FRAUD"

    else:

        result = "LEGITIMATE"

    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return {

        "transaction_id":
            data["transaction_id"],

        "customer_id":
            data["customer_id"],

        "fraud_probability":
            round(
                float(probability),
                4
            ),

        "threshold":
            threshold,

        "prediction":
            result,

        "is_fraud":
            prediction
    }