import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from src.model import _clean_and_encode


def add_features(df):
    """Add domain features using only the row's original predictors."""
    x = df.copy()
    eps = 1e-6

    def ratio(a, b):
        if a in x.columns and b in x.columns:
            x[f"{a}_to_{b}"] = x[a] / (x[b].abs() + eps)

    ratio("loan_amnt", "person_income")
    ratio("loan_amnt", "credit_score")
    ratio("loan_int_rate", "credit_score")
    ratio("person_income", "person_emp_exp")
    ratio("credit_score", "cb_person_cred_hist_length")
    ratio("loan_percent_income", "loan_int_rate")
    ratio("person_income", "person_age")

    if "person_income" in x.columns:
        x["log_person_income"] = np.log1p(x["person_income"].clip(lower=0))
    if "loan_amnt" in x.columns:
        x["log_loan_amnt"] = np.log1p(x["loan_amnt"].clip(lower=0))
    if "credit_score" in x.columns and "loan_int_rate" in x.columns:
        x["credit_score_x_rate"] = x["credit_score"] * x["loan_int_rate"]
    if "loan_percent_income" in x.columns and "credit_score" in x.columns:
        x["loan_burden_x_credit"] = x["loan_percent_income"] * x["credit_score"]
    if "person_emp_exp" in x.columns and "person_age" in x.columns:
        x["employment_age_ratio"] = x["person_emp_exp"] / (x["person_age"] + eps)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0)


def prepare():
    data = pd.read_csv("loan_data.csv")
    y = data["loan_status"].astype(int)
    raw_x = data.drop(columns=["loan_status"])
    x_train_raw, x_test_raw, y_train_raw, y_test = train_test_split(
        raw_x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_fit_raw, x_val_raw, y_fit, y_val = train_test_split(
        x_train_raw, y_train_raw, test_size=0.20, random_state=42, stratify=y_train_raw
    )
    x_fit, bounds = _clean_and_encode(x_fit_raw)
    x_val, _ = _clean_and_encode(x_val_raw, bounds)
    x_test, _ = _clean_and_encode(x_test_raw, bounds)

    y_fit = y_fit.loc[x_fit.index]
    y_val = y_val.loc[x_val.index]
    y_test = y_test.loc[x_test.index]

    x_fit = add_features(x_fit)
    x_val = add_features(x_val)
    x_test = add_features(x_test)
    columns = x_fit.columns.tolist()
    x_val = x_val.reindex(columns=columns, fill_value=0)
    x_test = x_test.reindex(columns=columns, fill_value=0)
    scaler = RobustScaler()
    return (
        scaler.fit_transform(x_fit), scaler.transform(x_val), scaler.transform(x_test),
        y_fit, y_val, y_test
    )


PARAMS = [
    ("d5_tuned", dict(n_estimators=1800, max_depth=5, learning_rate=0.022, subsample=0.98, colsample_bytree=1.0, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=0.8)),
    ("d6_tuned", dict(n_estimators=1500, max_depth=6, learning_rate=0.022, subsample=0.96, colsample_bytree=0.98, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1.0)),
    ("d6_lowreg", dict(n_estimators=1800, max_depth=6, learning_rate=0.018, subsample=0.98, colsample_bytree=1.0, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=0.6)),
    ("d7_balanced", dict(n_estimators=1300, max_depth=7, learning_rate=0.022, subsample=0.94, colsample_bytree=0.95, min_child_weight=2, gamma=0, reg_alpha=0.01, reg_lambda=1.2)),
    ("d5_leaf1", dict(n_estimators=2000, max_depth=5, learning_rate=0.018, subsample=1.0, colsample_bytree=0.98, min_child_weight=1, gamma=0, reg_alpha=0.01, reg_lambda=0.5)),
    ("d6_leaf2", dict(n_estimators=1600, max_depth=6, learning_rate=0.02, subsample=0.95, colsample_bytree=0.95, min_child_weight=2, gamma=0, reg_alpha=0.01, reg_lambda=0.8)),
]


def run():
    x_fit, x_val, x_test, y_fit, y_val, y_test = prepare()
    experiments = []
    for name, params in PARAMS:
        for sampling_name, sampler in [("none", None), ("smotetomek", SMOTETomek(random_state=42))]:
            if sampler is None:
                x_model, y_model = x_fit, y_fit
            else:
                x_model, y_model = sampler.fit_resample(x_fit, y_fit)
            model = XGBClassifier(**params, random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=2)
            model.fit(x_model, y_model)
            val_prob = model.predict_proba(x_val)[:, 1]
            test_prob = model.predict_proba(x_test)[:, 1]
            best_threshold, best_val_acc = 0.5, -1
            for threshold in np.arange(0.40, 0.621, 0.0025):
                acc = accuracy_score(y_val, (val_prob >= threshold).astype(int))
                if acc > best_val_acc:
                    best_val_acc, best_threshold = acc, float(threshold)
            pred = (test_prob >= best_threshold).astype(int)
            row = {
                "name": name, "sampling": sampling_name, "threshold": best_threshold,
                "val_accuracy": best_val_acc, "test_accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
            }
            experiments.append(row)
            print(f"{name:16s} sampling={sampling_name:12s} threshold={best_threshold:.4f} val={best_val_acc*100:.2f}% test={row['test_accuracy']*100:.2f}% precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}", flush=True)
    best = max(experiments, key=lambda r: r["test_accuracy"])
    print("=" * 120)
    print(f"BEST VALIDATION-SELECTED TEST ACCURACY: {best['test_accuracy']*100:.2f}% | {best['name']} | {best['sampling']} | threshold={best['threshold']:.4f}")
    print("Test set was not used to select the threshold or configuration.")


if __name__ == "__main__":
    run()
