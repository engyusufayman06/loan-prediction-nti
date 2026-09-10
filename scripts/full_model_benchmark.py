from __future__ import annotations

import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.model import XGB_CONFIG, _clean_and_encode


def prepare():
    data = pd.read_csv("loan_data.csv")
    y = data["loan_status"].astype(int)
    raw_x = data.drop(columns=["loan_status"])
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        raw_x, y, test_size=0.20, random_state=42, stratify=y
    )

    x_train, bounds = _clean_and_encode(x_train_raw)
    x_test, _ = _clean_and_encode(x_test_raw, bounds)
    y_train = y_train_raw.loc[x_train.index]
    y_test = y_test_raw.loc[x_test.index]

    columns = x_train.columns.tolist()
    x_test = x_test.reindex(columns=columns, fill_value=0)

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    sampler = SMOTETomek(random_state=42)
    x_train, y_train = sampler.fit_resample(x_train, y_train)
    return x_train, x_test, y_train, y_test


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=2),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features="sqrt", n_jobs=2, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features="sqrt", n_jobs=2, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=42),
        "SVM": SVC(C=2.0, kernel="rbf", gamma="scale", random_state=42),
        "XGBoost": XGBClassifier(**XGB_CONFIG),
    }


def run():
    x_train, x_test, y_train, y_test = prepare()
    print(f"TRAIN_ROWS={len(y_train)} TEST_ROWS={len(y_test)} FEATURES={x_train.shape[1]}", flush=True)
    results = []

    for name, model in build_models().items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(x_train, y_train)
        pred = model.predict(x_test).astype(int)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        results.append(row)
        print(
            f"{name:24s} accuracy={row['accuracy']*100:.2f}% "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}",
            flush=True,
        )

    best = max(results, key=lambda r: r["accuracy"])
    print("\n" + "=" * 100)
    print("FULL MODEL BENCHMARK")
    print("=" * 100)
    print("Model | Accuracy | Precision | Recall | F1")
    for r in sorted(results, key=lambda r: r["accuracy"], reverse=True):
        print(f"{r['model']} | {r['accuracy']*100:.2f}% | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f}")
    print("=" * 100)
    print(f"BEST={best['model']} ACCURACY={best['accuracy']*100:.2f}%")


if __name__ == "__main__":
    run()
