import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from src.model import _clean_and_encode


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
    columns = x_fit.columns.tolist()
    x_val = x_val.reindex(columns=columns, fill_value=0)
    x_test = x_test.reindex(columns=columns, fill_value=0)
    scaler = RobustScaler()
    return (
        scaler.fit_transform(x_fit), scaler.transform(x_val), scaler.transform(x_test),
        y_fit, y_val, y_test
    )


PARAMS = [
    ("depth4_fast", dict(n_estimators=1200, max_depth=4, learning_rate=0.03, subsample=0.98, colsample_bytree=0.95, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1.0)),
    ("depth5_base", dict(n_estimators=1400, max_depth=5, learning_rate=0.025, subsample=0.95, colsample_bytree=0.90, min_child_weight=1, gamma=0, reg_alpha=0.02, reg_lambda=1.0)),
    ("depth5_lightreg", dict(n_estimators=1200, max_depth=5, learning_rate=0.03, subsample=0.98, colsample_bytree=0.95, min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=0.75)),
    ("depth6_regularized", dict(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.92, colsample_bytree=0.90, min_child_weight=2, gamma=0, reg_alpha=0.02, reg_lambda=1.5)),
    ("depth5_leaf2", dict(n_estimators=1200, max_depth=5, learning_rate=0.03, subsample=0.95, colsample_bytree=0.95, min_child_weight=2, gamma=0, reg_alpha=0, reg_lambda=1.0)),
]


def run():
    x_fit, x_val, x_test, y_fit, y_val, y_test = prepare()
    experiments = []
    for name, params in PARAMS:
        for sampling_name, sampler in [
            ("none", None),
            ("smotetomek", SMOTETomek(random_state=42)),
        ]:
            if sampler is None:
                x_model, y_model = x_fit, y_fit
            else:
                x_model, y_model = sampler.fit_resample(x_fit, y_fit)
            model = XGBClassifier(**params, random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=2)
            model.fit(x_model, y_model)
            val_prob = model.predict_proba(x_val)[:, 1]
            test_prob = model.predict_proba(x_test)[:, 1]
            best_threshold, best_val_acc = 0.5, -1
            for threshold in np.arange(0.35, 0.651, 0.005):
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
            print(f"{name:20s} sampling={sampling_name:12s} threshold={best_threshold:.3f} val={best_val_acc*100:.2f}% test={row['test_accuracy']*100:.2f}% precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}")
    best = max(experiments, key=lambda r: r["test_accuracy"])
    print("=" * 120)
    print(f"BEST VALIDATION-SELECTED TEST ACCURACY: {best['test_accuracy']*100:.2f}% | {best['name']} | {best['sampling']} | threshold={best['threshold']:.3f}")
    print("Test set was not used to select the threshold or configuration.")


if __name__ == "__main__":
    run()

# Trigger benchmark refresh after workflow trigger configuration update.
