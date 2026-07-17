import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def run_shap_analysis(
    model,
    X_train,
    X_test,
    output_folder="figures"
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    print("Generating SHAP explanations...")


    # Convert to DataFrame
    X_train_df = pd.DataFrame(
        X_train,
        columns=X_train.columns
    )

    X_test_df = pd.DataFrame(
        X_test,
        columns=X_test.columns
    )


    # XGBoost Tree Explainer
    explainer = shap.TreeExplainer(
        model
    )


    sample_size = min(
        1000,
        len(X_test_df)
    )

    X_sample = X_test_df.iloc[:sample_size]


    shap_values = explainer.shap_values(
        X_sample
    )


    # SHAP Summary Plot

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # SHAP Feature Importance

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # Feature importance table

    importance = pd.DataFrame(
        {
            "Feature": X_sample.columns,
            "Mean_Abs_SHAP":
            np.abs(shap_values).mean(axis=0)
        }
    )


    importance = importance.sort_values(
        "Mean_Abs_SHAP",
        ascending=False
    )


    importance.to_csv(
        f"{output_folder}/shap_feature_importance.csv",
        index=False
    )


    print(
        "SHAP analysis completed"
    )

    return importance
