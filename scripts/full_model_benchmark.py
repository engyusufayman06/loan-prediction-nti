from __future__ import annotations

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def prepare():
    data = pd.read_csv("loan_data.csv")
    data = data.dropna(subset=["loan_status"]).copy()
    data = data[
        (data["person_age"] <= 80)
        & (data["person_emp_exp"] <= 50)
    ].copy()

    data["income_to_loan_ratio"] = data["person_income"] / (data["loan_amnt"] + 1)
    data["emp_length_to_age_ratio"] = data["person_emp_exp"] / (data["person_age"] + 1)
    data["cred_hist_to_age_ratio"] = data["cb_person_cred_hist_length"] / (
        data["person_age"] + 1
    )
    data["high_risk_income_ratio"] = (
        data["loan_percent_income"] * data["loan_int_rate"]
    )
    data["is_renting"] = (data["person_home_ownership"] == "RENT").astype(int)

    X = data.drop(columns=["loan_status"])
    y = data["loan_status"].astype(int)

    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    return X, X_train_smote, X_test, X_train_scaled, X_test_scaled, y_train_smote, y_test


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=42, max_iter=300
        ),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def run():
    X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = prepare()
    needs_scaling = {"Logistic Regression", "KNN", "SVM"}

    results = []
    for name, model in build_models().items():
        if name in needs_scaling:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1-Score": f1_score(y_test, pred),
                "ROC-AUC": roc_auc_score(y_test, prob),
            }
        )

    benchmark = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    print(benchmark.to_string(index=False))
    print()
    print(
        f"Rows: raw=45000 clean={len(X)} train={len(X) - len(y_test)} "
        f"test={len(y_test)} smote_train={len(y_train)} features={X.shape[1]}"
    )


if __name__ == "__main__":
    run()
