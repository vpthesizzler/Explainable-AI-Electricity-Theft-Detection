import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def tune_threshold(y_true,y_prob):

    thresholds=np.arange(
        0.05,
        0.9,
        0.05
    )

    best_threshold=0.5
    best_f1=0


    for t in thresholds:

        y_pred=(y_prob>=t).astype(int)

        f1=f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if f1>best_f1:

            best_f1=f1
            best_threshold=t


    return best_threshold



def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    y_prob=model.predict_proba(
        X_test
    )[:,1]


    threshold=tune_threshold(
        y_test,
        y_prob
    )


    y_pred=(
        y_prob>=threshold
    ).astype(int)


    results={

        "Model":name,

        "Threshold":threshold,

        "Accuracy":
        accuracy_score(
            y_test,
            y_pred
        ),

        "Precision":
        precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall":
        recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1":
        f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC":
        roc_auc_score(
            y_test,
            y_prob
        )
    }


    return results
