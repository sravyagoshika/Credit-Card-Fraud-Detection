import pandas as pd
import numpy as np
import os
import joblib
import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("=" * 70)

# ============================================================
# 1. Load SMOTE training data
# ============================================================

X_train = pd.read_csv(
    "data/processed/smote/X_train_smote.csv"
)

y_train = pd.read_csv(
    "data/processed/smote/y_train_smote.csv"
)["is_fraud"]

# ============================================================
# 2. Load original test data
# ============================================================

X_test = pd.read_csv(
    "data/processed/smote/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/smote/y_test.csv"
)["is_fraud"]

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ============================================================
# 3. Define models
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
}

# ============================================================
# 4. Create model directory
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

# ============================================================
# 5. Train models
# ============================================================

trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 70)

    print(f"Training: {name}")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    trained_models[name] = model

    # Safe filename
    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        f"models/{filename}"
    )

    print(f"{name} trained successfully!")

    print(
        f"Training time: {training_time:.2f} seconds"
    )

# ============================================================
# 6. Save feature names
# ============================================================

with open(
    "models/feature_names.txt",
    "w"
) as file:

    for column in X_train.columns:
        file.write(column + "\n")

# ============================================================
# 7. Final message
# ============================================================

print("\n" + "=" * 70)

print("ALL MODELS TRAINED SUCCESSFULLY!")

print("\nModels saved in:")
print("models/")

print("\nSaved models:")

for name in models.keys():

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".pkl"
    )

    print(" -", filename)

print("=" * 70)