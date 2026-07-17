import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data(
    sgcc_path="sgcc_ml_ready.csv",
    pmu_path="merged_frequency_data.csv"
):

    sgcc = pd.read_csv(sgcc_path)
    pmu = pd.read_csv(pmu_path)

    print("SGCC shape:", sgcc.shape)
    print("PMU shape:", pmu.shape)


    y = sgcc["label"]


    sgcc_features = sgcc.drop(
        columns=["label"],
        errors="ignore"
    )

    pmu_features = pmu.drop(
        columns=["label"],
        errors="ignore"
    )


    X = pd.concat(
        [
            sgcc_features.reset_index(drop=True),
            pmu_features.reset_index(drop=True)
        ],
        axis=1
    )


    X = X.loc[:, ~X.columns.duplicated()]

    X = X.select_dtypes(
        include=[np.number]
    )


    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )
