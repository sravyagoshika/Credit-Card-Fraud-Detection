import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION")
print("=" * 70)

# ============================================================
# 1. Load test data
# ============================================================

X_test = pd.read_csv(
    "data/processed/smote/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/smote/y_test.csv"
)["is_fraud"]

print("\nTesting data:")
print(X_test.shape)

print("\nTest class distribution:")
print(y_test.value_counts())

# ============================================================
# 2. Define models
# ============================================================

model_files = {
    "Logistic Regression":
        "models/logistic_regression.pkl",

    "Decision Tree":
        "models/decision_tree.pkl",

    "Random Forest":
        "models/random_forest.pkl",

    "Gradient Boosting":
        "models/gradient_boosting.pkl",

    "XGBoost":
        "models/xgboost.pkl",

    "LightGBM":
        "models/lightgbm.pkl"
}

# ============================================================
# 3. Create output directories
# ============================================================

os.makedirs(
    "reports/figures",
    exist_ok=True
)

os.makedirs(
    "reports/metrics",
    exist_ok=True
)

# ============================================================
# 4. Evaluate models
# ============================================================

results = []

for name, path in model_files.items():

    print("\n" + "-" * 70)

    print(f"Evaluating: {name}")

    # Load model
    model = joblib.load(path)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of fraud
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    # --------------------------------------------------------
    # Save confusion matrix figure
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Fraud"
        ]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {name}"
    )

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        f"reports/figures/"
        f"confusion_matrix_{filename}.png",
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 5. Create comparison table
# ============================================================

results_df = pd.DataFrame(results)

# Sort by F1 score
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# 6. Save comparison results
# ============================================================

results_df.to_csv(
    "reports/metrics/model_comparison.csv",
    index=False
)

# ============================================================
# 7. Identify best model
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Precision: {best_model['Precision']:.4f}"
)

print(
    f"Recall: {best_model['Recall']:.4f}"
)

print(
    f"F1 Score: {best_model['F1 Score']:.4f}"
)

print(
    f"ROC-AUC: {best_model['ROC-AUC']:.4f}"
)

print(
    f"PR-AUC: {best_model['PR-AUC']:.4f}"
)

print("\nEvaluation completed successfully!")

print("\nResults saved to:")
print("reports/metrics/model_comparison.csv")

print("\nConfusion matrices saved to:")
print("reports/figures/")

print("=" * 70)