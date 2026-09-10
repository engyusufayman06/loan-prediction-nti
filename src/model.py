from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

NUMERIC_CLIP_COLUMNS = [
    "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
]

MODEL_INPUT_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_exp",
    "loan_intent", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file",
    "person_gender", "person_education",
]

# Selected from a reproducible XGBoost benchmark on the fixed stratified split.
# This configuration gave the strongest measured accuracy among the tested
# candidates while keeping regularization and subsampling to limit overfitting.
XGB_CONFIG = {
    "n_estimators": 900,
    "max_depth": 5,
    "learning_rate": 0.035,
    "subsample": 0.95,
    "colsample_bytree": 0.90,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.02,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 2,
}


@dataclass
class LoanModelBundle:
    model: Any
    scaler: RobustScaler
    feature_columns: list[str]
    clip_bounds: dict[str, tuple[float, float]]


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _clean_and_encode(df: pd.DataFrame, clip_bounds=None):
    data = df.copy()
    if "loan_id" in data.columns:
        data = data.drop(columns=["loan_id"])

    if clip_bounds is None:
        clip_bounds = {
            c: _iqr_bounds(data[c])
            for c in NUMERIC_CLIP_COLUMNS
            if c in data.columns
        }

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

    categorical = [c for c in ["person_home_ownership", "loan_intent"] if c in data.columns]
    if categorical:
        data = pd.get_dummies(data, columns=categorical, drop_first=True, dtype=float)

    return data, clip_bounds


def train_model(data_path: str | Path) -> tuple[LoanModelBundle, dict[str, Any]]:
    data = pd.read_csv(data_path)
    if "loan_status" not in data.columns:
        raise ValueError("Dataset must contain a 'loan_status' target column.")

    y = data["loan_status"].astype(int)
    raw_x = data.drop(columns=["loan_status"])

    # Stratification makes the held-out test distribution representative of
    # the original target distribution.
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        raw_x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Fit preprocessing decisions from training data only to avoid test-set
    # leakage, then reuse the exact same feature contract everywhere.
    x_train, clip_bounds = _clean_and_encode(x_train_raw)
    x_test, _ = _clean_and_encode(x_test_raw, clip_bounds)
    y_train = y_train_raw.loc[x_train.index]
    y_test = y_test_raw.loc[x_test.index]

    feature_columns = x_train.columns.tolist()
    x_test = x_test.reindex(columns=feature_columns, fill_value=0)

    # RobustScaler is retained because SMOTETomek relies on neighborhood
    # distances; it also keeps the inference contract backward-compatible.
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    sampler = SMOTETomek(random_state=42)
    x_train_balanced, y_train_balanced = sampler.fit_resample(x_train_scaled, y_train)

    model = XGBClassifier(**XGB_CONFIG)
    model.fit(x_train_balanced, y_train_balanced)

    y_pred = model.predict(x_test_scaled).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "train_rows_after_smotetomek": int(len(y_train_balanced)),
        "test_rows": int(len(y_test)),
        "feature_count": int(len(feature_columns)),
        "model_config": XGB_CONFIG.copy(),
    }
    return LoanModelBundle(model, scaler, feature_columns, clip_bounds), metrics


def _prepare_for_inference(bundle: LoanModelBundle, data: pd.DataFrame) -> pd.DataFrame:
    encoded, _ = _clean_and_encode(data, bundle.clip_bounds)
    return encoded.reindex(columns=bundle.feature_columns, fill_value=0)


def predict_one(bundle: LoanModelBundle, applicant: dict[str, Any]) -> dict[str, Any]:
    encoded = _prepare_for_inference(bundle, pd.DataFrame([applicant]))
    scaled = bundle.scaler.transform(encoded)
    prediction = int(bundle.model.predict(scaled)[0])
    probabilities = bundle.model.predict_proba(scaled)[0]
    return {
        "prediction": prediction,
        "approval_probability": float(probabilities[1]),
        "rejection_probability": float(probabilities[0]),
        "label": "Approved" if prediction == 1 else "Rejected",
    }


def predict_batch(bundle: LoanModelBundle, data: pd.DataFrame) -> pd.DataFrame:
    """Run the exact trained inference pipeline over a dataframe."""
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    source = data.copy()
    encoded = _prepare_for_inference(bundle, source)
    valid_source = source.loc[encoded.index]
    row_ids = (
        valid_source["loan_id"]
        if "loan_id" in valid_source.columns
        else pd.Series(range(1, len(valid_source) + 1), index=valid_source.index, name="loan_id")
    )

    scaled = bundle.scaler.transform(encoded)
    predictions = bundle.model.predict(scaled).astype(int)
    probabilities = bundle.model.predict_proba(scaled)

    result = pd.DataFrame(
        {
            "loan_id": row_ids.values,
            "prediction": predictions,
            "predicted_status": ["Approved" if p == 1 else "Rejected" for p in predictions],
            "approval_probability": probabilities[:, 1],
            "rejection_probability": probabilities[:, 0],
        },
        index=valid_source.index,
    )

    if "loan_status" in valid_source.columns:
        result["actual_status"] = valid_source["loan_status"].values
        result["correct"] = result["prediction"] == result["actual_status"].astype(int)

    return result.reset_index(drop=True)


def save_bundle(bundle: LoanModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> LoanModelBundle:
    return joblib.load(path)
