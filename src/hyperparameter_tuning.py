import pandas as pd
import numpy as np
import os
import joblib
import time

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - HYPERPARAMETER TUNING")
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
# 2. Load untouched test data
# ============================================================

X_test = pd.read_csv(
    "data/processed/smote/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/smote/y_test.csv"
)["is_fraud"]


print("\nTraining shape:")
print(X_train.shape)

print("\nTesting shape:")
print(X_test.shape)


# ============================================================
# 3. Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. Decision Tree
# ============================================================

print("\n" + "-" * 70)
print("Tuning Decision Tree...")
print("-" * 70)

dt = DecisionTreeClassifier(
    random_state=42
)

dt_params = {
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"]
}

dt_search = RandomizedSearchCV(
    dt,
    dt_params,
    n_iter=15,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

start = time.time()

dt_search.fit(
    X_train,
    y_train
)

print(
    f"Time taken: {time.time() - start:.2f} seconds"
)

print("\nBest Decision Tree parameters:")
print(dt_search.best_params_)

print(
    f"Best CV F1 Score: "
    f"{dt_search.best_score_:.4f}"
)


# ============================================================
# 5. Logistic Regression
# ============================================================

print("\n" + "-" * 70)
print("Tuning Logistic Regression...")
print("-" * 70)

lr = LogisticRegression(
    max_iter=2000,
    random_state=42
)

lr_params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
    "class_weight": [None, "balanced"]
}

lr_search = RandomizedSearchCV(
    lr,
    lr_params,
    n_iter=10,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

start = time.time()

lr_search.fit(
    X_train,
    y_train
)

print(
    f"Time taken: {time.time() - start:.2f} seconds"
)

print("\nBest Logistic Regression parameters:")
print(lr_search.best_params_)

print(
    f"Best CV F1 Score: "
    f"{lr_search.best_score_:.4f}"
)


# ============================================================
# 6. XGBoost
# ============================================================

print("\n" + "-" * 70)
print("Tuning XGBoost...")
print("-" * 70)

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb,
    xgb_params,
    n_iter=15,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

start = time.time()

xgb_search.fit(
    X_train,
    y_train
)

print(
    f"Time taken: {time.time() - start:.2f} seconds"
)

print("\nBest XGBoost parameters:")
print(xgb_search.best_params_)

print(
    f"Best CV F1 Score: "
    f"{xgb_search.best_score_:.4f}"
)


# ============================================================
# 7. LightGBM
# ============================================================

print("\n" + "-" * 70)
print("Tuning LightGBM...")
print("-" * 70)

lgbm = LGBMClassifier(
    random_state=42,
    verbose=-1
)

lgbm_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [-1, 3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 50, 70],
    "subsample": [0.7, 0.8, 1.0]
}

lgbm_search = RandomizedSearchCV(
    lgbm,
    lgbm_params,
    n_iter=15,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

start = time.time()

lgbm_search.fit(
    X_train,
    y_train
)

print(
    f"Time taken: {time.time() - start:.2f} seconds"
)

print("\nBest LightGBM parameters:")
print(lgbm_search.best_params_)

print(
    f"Best CV F1 Score: "
    f"{lgbm_search.best_score_:.4f}"
)


# ============================================================
# 8. Save tuned models
# ============================================================

os.makedirs(
    "models/tuned",
    exist_ok=True
)

joblib.dump(
    dt_search.best_estimator_,
    "models/tuned/decision_tree_tuned.pkl"
)

joblib.dump(
    lr_search.best_estimator_,
    "models/tuned/logistic_regression_tuned.pkl"
)

joblib.dump(
    xgb_search.best_estimator_,
    "models/tuned/xgboost_tuned.pkl"
)

joblib.dump(
    lgbm_search.best_estimator_,
    "models/tuned/lightgbm_tuned.pkl"
)


# ============================================================
# 9. Test-set comparison
# ============================================================

tuned_models = {

    "Decision Tree": dt_search.best_estimator_,

    "Logistic Regression":
        lr_search.best_estimator_,

    "XGBoost":
        xgb_search.best_estimator_,

    "LightGBM":
        lgbm_search.best_estimator_
}


results = []


print("\n" + "=" * 70)
print("TUNED MODEL TEST PERFORMANCE")
print("=" * 70)


for name, model in tuned_models.items():

    y_pred = model.predict(
        X_test
    )

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

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n" + name)

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )


# ============================================================
# 10. Save tuning results
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    "reports/metrics/tuned_model_comparison.csv",
    index=False
)


print("\n" + "=" * 70)

print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")

print("\nTuned models saved in:")
print("models/tuned/")

print("\nResults saved in:")
print("reports/metrics/tuned_model_comparison.csv")

print("=" * 70)