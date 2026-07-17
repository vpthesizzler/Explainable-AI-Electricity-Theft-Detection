import pandas as pd
import os

from preprocessing import load_and_prepare_data
from train_model import build_models, train_models
from evaluate import evaluate_model
from explainability import run_shap_analysis


# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading datasets...")

X_train, X_test, y_train, y_test = load_and_prepare_data()


print("\nTraining data shape:")
print(X_train.shape)



# =====================================================
# BUILD AND TRAIN MODELS
# =====================================================

models = build_models(
    y_train
)


models = train_models(
    models,
    X_train,
    y_train
)



# =====================================================
# MODEL EVALUATION
# =====================================================

results = []


for name, model in models.items():

    print("\nEvaluating:", name)

    result = evaluate_model(
        name,
        model,
        X_test,
        y_test
    )

results.append(result)



results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    "results/model_comparison_threshold.csv",
    index=False
)


print("\nModel Results:")
print(results_df)



# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

print("\nRunning SHAP analysis...")


# XGBoost selected for SHAP interpretation
xgb_model = models["XGBoost"]


run_shap_analysis(
    xgb_model,
    X_train,
    X_test,
    "figures"
)


print("\n===================================")
print("COMPLETE PIPELINE FINISHED")
print("Results saved in results/")
print("Figures saved in figures/")
print("===================================")
