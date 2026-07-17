from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


def build_models(y_train):

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    scale_pos_weight = (
        len(y_train[y_train == 0]) /
        max(len(y_train[y_train == 1]),1)
    )


    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )


    hybrid = VotingClassifier(
        estimators=[
            ("rf",rf),
            ("xgb",xgb)
        ],
        voting="soft",
        weights=[1,2]
    )


    return {
        "Random Forest":rf,
        "XGBoost":xgb,
        "Hybrid":hybrid
    }



def train_models(models,X_train,y_train):

    for name,model in models.items():

        print("Training:",name)

        model.fit(
            X_train,
            y_train
        )

    return models
