from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


NUMERIC_CLIP_COLUMNS = [
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]


@dataclass
class LoanModelBundle:
    model: Any
    scaler: RobustScaler
    feature_columns: list[str]
    clip_bounds: dict[str, tuple[float, float]]


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _clean_and_encode(
    df: pd.DataFrame,
    clip_bounds: dict[str, tuple[float, float]] | None = None,
):
    data = df.copy()

    if "loan_id" in data.columns:
        data = data.drop(columns=["loan_id"])

    if clip_bounds is None:
        clip_bounds = {}
        for col in NUMERIC_CLIP_COLUMNS:
            if col in data.columns:
                clip_bounds[col] = _iqr_bounds(data[col])

    for col, (low, high) in clip_bounds.items():
        if col in data.columns:
            data[col] = data[col].clip(lower=low, upper=high)

    if "person_age" in data.columns:
        data = data.loc[data["person_age"] <= 80].copy()
    if "person_emp_exp" in data.columns:
        data = data.loc[data["person_emp_exp"] <= 60].copy()

    mappings = {
        "person_gender": {"male": 1, "female": 0},
        "previous_loan_defaults_on_file": {"Yes": 1, "No": 0},
        "person_education": {
            "High School": 1,
            "Associate": 2,
            "Bachelor": 3,
            "Master": 4,
            "Doctorate": 5,
        },
    }

    for col, mapping in mappings.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    categorical = [
        c
        for c in ["person_home_ownership", "loan_intent"]
        if c in data.columns
    ]
    if categorical:
        data = pd.get_dummies(data, columns=categorical, drop_first=True)

    return data, clip_bounds


def train_model(data_path: str | Path) -> tuple[LoanModelBundle, dict[str, Any]]:
    data = pd.read_csv(data_path)

    if "loan_status" not in data.columns:
        raise ValueError("Dataset must contain a 'loan_status' target column.")

    y = data["loan_status"].astype(int)
    X_raw = data.drop(columns=["loan_status"])

    X_clean, clip_bounds = _clean_and_encode(X_raw)
    y = y.loc[X_clean.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.20,
        random_state=42,
        stratify=None,
    )

    feature_columns = X_train.columns.tolist()

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sampler = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = sampler.fit_resample(
        X_train_scaled, y_train
    )

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "train_rows_after_smotetomek": int(len(y_train_balanced)),
        "test_rows": int(len(y_test)),
    }

    bundle = LoanModelBundle(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        clip_bounds=clip_bounds,
    )
    return bundle, metrics


def predict_one(bundle: LoanModelBundle, applicant: dict[str, Any]) -> dict[str, Any]:
    row = pd.DataFrame([applicant])
    encoded, _ = _clean_and_encode(row, bundle.clip_bounds)
    encoded = encoded.reindex(columns=bundle.feature_columns, fill_value=0)
    scaled = bundle.scaler.transform(encoded)

    prediction = int(bundle.model.predict(scaled)[0])
    probabilities = bundle.model.predict_proba(scaled)[0]

    return {
        "prediction": prediction,
        "approval_probability": float(probabilities[1]),
        "rejection_probability": float(probabilities[0]),
        "label": "Approved" if prediction == 1 else "Rejected",
    }


def save_bundle(bundle: LoanModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> LoanModelBundle:
    return joblib.load(path)
