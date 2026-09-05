import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE


print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - THRESHOLD TUNING")
print("=" * 70)


# ============================================================
# 1. Load ORIGINAL training data
# ============================================================

X_train_full = pd.read_csv(
    "data/processed/X_train.csv"
)

y_train_full = pd.read_csv(
    "data/processed/y_train.csv"
)["is_fraud"]


# ============================================================
# 2. Load ORIGINAL untouched test data
# ============================================================

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
)["is_fraud"]


print("\nOriginal training shape:")
print(X_train_full.shape)

print("\nTest shape:")
print(X_test.shape)


# ============================================================
# 3. Split original training data
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=42
)


print("\nTraining portion:")
print(X_train.shape)

print("\nValidation portion:")
print(X_validation.shape)

print("\nValidation class distribution:")
print(y_validation.value_counts())


# ============================================================
# 4. Apply SMOTE ONLY to training portion
# ============================================================

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nAfter SMOTE:")
print(X_train_smote.shape)

print("\nSMOTE class distribution:")
print(y_train_smote.value_counts())


# ============================================================
# 5. Load tuned models
# ============================================================

model_paths = {

    "Decision Tree":
        "models/tuned/decision_tree_tuned.pkl",

    "Logistic Regression":
        "models/tuned/logistic_regression_tuned.pkl",

    "XGBoost":
        "models/tuned/xgboost_tuned.pkl",

    "LightGBM":
        "models/tuned/lightgbm_tuned.pkl"
}


# ============================================================
# 6. Threshold values
# ============================================================

thresholds = np.arange(
    0.05,
    0.51,
    0.05
)


all_results = []


# ============================================================
# 7. Train and tune threshold for each model
# ============================================================

for name, path in model_paths.items():

    print("\n" + "-" * 70)
    print(f"Processing: {name}")
    print("-" * 70)

    # Load previously tuned model
    saved_model = joblib.load(path)

    # Clone model so we retrain it correctly
    model = clone(saved_model)

    print("Training model on SMOTE training data...")

    model.fit(
        X_train_smote,
        y_train_smote
    )

    # Validation probabilities
    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    print("\nThreshold results:")

    best_threshold = 0.50
    best_f1 = -1
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:

        validation_prediction = (
            validation_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            validation_prediction,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            validation_prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            validation_prediction,
            zero_division=0
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    print("\nBest validation threshold:")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Precision : {best_precision:.4f}")
    print(f"Recall    : {best_recall:.4f}")
    print(f"F1 Score  : {best_f1:.4f}")


    # ========================================================
    # 8. Evaluate on untouched test data
    # ========================================================

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_prediction = (
        test_probability >= best_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    test_cm = confusion_matrix(
        y_test,
        test_prediction
    )


    print("\nFINAL TEST PERFORMANCE:")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")

    print("\nConfusion Matrix:")
    print(test_cm)


    all_results.append({

        "Model": name,

        "Best Threshold":
            best_threshold,

        "Validation Precision":
            best_precision,

        "Validation Recall":
            best_recall,

        "Validation F1":
            best_f1,

        "Test Precision":
            test_precision,

        "Test Recall":
            test_recall,

        "Test F1":
            test_f1
    })


# ============================================================
# 9. Create results table
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    by="Test F1",
    ascending=False
)


# ============================================================
# 10. Save results
# ============================================================

os.makedirs(
    "reports/metrics",
    exist_ok=True
)

results_df.to_csv(
    "reports/metrics/threshold_comparison.csv",
    index=False
)


# ============================================================
# 11. Display final comparison
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD TUNING COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. Best final model
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL AFTER THRESHOLD TUNING")
print("=" * 70)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Threshold: "
    f"{best_model['Best Threshold']:.2f}"
)

print(
    f"Test Precision: "
    f"{best_model['Test Precision']:.4f}"
)

print(
    f"Test Recall: "
    f"{best_model['Test Recall']:.4f}"
)

print(
    f"Test F1 Score: "
    f"{best_model['Test F1']:.4f}"
)

print("\nThreshold tuning completed successfully!")

print("\nResults saved to:")
print(
    "reports/metrics/threshold_comparison.csv"
)

print("=" * 70)