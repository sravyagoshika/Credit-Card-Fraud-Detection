import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - MODEL EXPLAINABILITY")
print("=" * 70)

# ---------------------------------------------------------
# 1. Create output folders
# ---------------------------------------------------------

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/metrics", exist_ok=True)

# ---------------------------------------------------------
# 2. Load test data
# ---------------------------------------------------------

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

print("\nTest data shape:")
print(X_test.shape)

# ---------------------------------------------------------
# 3. Load Decision Tree model
# ---------------------------------------------------------

model_path = "models/decision_tree.pkl"

model = joblib.load(model_path)

print("\nModel loaded successfully:")
print("Decision Tree")

# ---------------------------------------------------------
# 4. Create SHAP explainer
# ---------------------------------------------------------

print("\nCreating SHAP explanations...")

explainer = shap.TreeExplainer(model)

# Use only 500 records for faster processing
sample_size = min(500, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

# ---------------------------------------------------------
# 5. Calculate SHAP values
# ---------------------------------------------------------

shap_output = explainer(X_sample)

print("SHAP values calculated successfully!")

# ---------------------------------------------------------
# 6. Extract SHAP values correctly
# ---------------------------------------------------------

shap_values = shap_output.values

print("\nSHAP values shape:")
print(shap_values.shape)

# ---------------------------------------------------------
# 7. Handle different SHAP output dimensions
# ---------------------------------------------------------

if shap_values.ndim == 3:

    # Binary classification output
    # Shape: samples × features × classes

    shap_values_plot = shap_values[:, :, 1]

elif shap_values.ndim == 2:

    # Normal output
    # Shape: samples × features

    shap_values_plot = shap_values

else:

    raise ValueError(
        f"Unexpected SHAP value shape: "
        f"{shap_values.shape}"
    )

print("\nSHAP values used for analysis:")
print(shap_values_plot.shape)

# ---------------------------------------------------------
# 8. SHAP Summary Plot
# ---------------------------------------------------------

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values_plot,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("reports/figures/shap_summary.png")

# ---------------------------------------------------------
# 9. Calculate Feature Importance
# ---------------------------------------------------------

print("\nCalculating feature importance...")

importance = np.abs(
    shap_values_plot
).mean(axis=0)

# Make sure importance is 1-dimensional
importance = np.asarray(
    importance
).reshape(-1)

feature_names = X_sample.columns.tolist()

print("\nNumber of features:")
print(len(feature_names))

print("Number of importance values:")
print(len(importance))

# ---------------------------------------------------------
# 10. Create feature importance DataFrame
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# ---------------------------------------------------------
# 11. Display top features
# ---------------------------------------------------------

print("\nTop 10 Important Features:")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 12. Save feature importance CSV
# ---------------------------------------------------------

feature_importance.to_csv(
    "reports/metrics/shap_feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("reports/metrics/shap_feature_importance.csv")

# ---------------------------------------------------------
# 13. Create Feature Importance Plot
# ---------------------------------------------------------

print("\nCreating feature importance plot...")

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Features Influencing Fraud Prediction"
)

plt.tight_layout()

plt.savefig(
    "reports/figures/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(
    "reports/figures/shap_feature_importance.png"
)

# ---------------------------------------------------------
# 14. Completion
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EXPLAINABILITY COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nGenerated files:")

print(
    " - reports/figures/shap_summary.png"
)

print(
    " - reports/figures/shap_feature_importance.png"
)

print(
    " - reports/metrics/shap_feature_importance.csv"
)

print("=" * 70)