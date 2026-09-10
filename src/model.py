from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


MODEL_INPUT_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_exp",
    "loan_intent",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "person_gender",
    "person_education",
]

ENGINEERED_FEATURES = [
    "income_to_loan_ratio",
    "emp_length_to_age_ratio",
    "cred_hist_to_age_ratio",
    "high_risk_income_ratio",
    "is_renting",
]

# This is the XGBoost configuration used in the notebook's nine-model benchmark.
# The notebook also contains a separate randomized-search experiment; the
# deployable app keeps the benchmark XGBoost contract so the documented metrics
# and inference path stay aligned.
XGB_CONFIG = {
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 2,
    "tree_method": "hist",
}


@dataclass
class LoanModelBundle:
    model: Any
    scaler: StandardScaler
    feature_columns: list[str]
    category_encoders: dict[str, LabelEncoder]


def _fit_encoders(data: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for col in data.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        encoder.fit(data[col].astype(str))
        encoders[col] = encoder
    return encoders


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()

    if "loan_id" in frame.columns:
        frame = frame.drop(columns=["loan_id"])

    frame = frame.loc[
        (frame["person_age"] <= 80) & (frame["person_emp_exp"] <= 50)
    ].copy()

    frame["income_to_loan_ratio"] = frame["person_income"] / (
        frame["loan_amnt"] + 1
    )
    frame["emp_length_to_age_ratio"] = frame["person_emp_exp"] / (
        frame["person_age"] + 1
    )
    frame["cred_hist_to_age_ratio"] = frame["cb_person_cred_hist_length"] / (
        frame["person_age"] + 1
    )
    frame["high_risk_income_ratio"] = (
        frame["loan_percent_income"] * frame["loan_int_rate"]
    )
    frame["is_renting"] = (
        frame["person_home_ownership"] == "RENT"
    ).astype(int)

    return frame


def _encode_with_fitted_encoders(
    data: pd.DataFrame,
    category_encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    frame = data.copy()

    for col, encoder in category_encoders.items():
        if col not in frame.columns:
            continue

        values = frame[col].astype(str)
        unknown = sorted(set(values) - set(encoder.classes_))
        if unknown:
            raise ValueError(
                f"Unknown category value(s) in '{col}': {', '.join(unknown)}"
            )
        frame[col] = encoder.transform(values)

    return frame


def _clean_and_encode(
    df: pd.DataFrame,
    category_encoders: dict[str, LabelEncoder] | None = None,
):
    """Apply the final notebook cleaning, feature engineering and encoding."""
    frame = _engineer_features(df)

    if category_encoders is None:
        category_encoders = _fit_encoders(frame)

    frame = _encode_with_fitted_encoders(frame, category_encoders)
    return frame, category_encoders


def train_model(data_path: str | Path) -> tuple[LoanModelBundle, dict[str, Any]]:
    data = pd.read_csv(data_path)
    if "loan_status" not in data.columns:
        raise ValueError("Dataset must contain a 'loan_status' target column.")

    data = data.dropna(subset=["loan_status"]).copy()
    y = data["loan_status"].astype(int)

    raw_x = data.drop(columns=["loan_status"])
    x, category_encoders = _clean_and_encode(raw_x)
    y = y.loc[x.index]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Match the final notebook: SMOTE is fitted only on the training split.
    smote = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    # Match the final notebook's scaling sequence.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_smote)
    x_test_scaled = scaler.transform(x_test)

    model = XGBClassifier(**XGB_CONFIG)
    model.fit(x_train_scaled, y_train_smote)

    y_pred = model.predict(x_test_scaled).astype(int)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "raw_rows": int(len(data)),
        "clean_rows": int(len(x)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_rows_after_smote": int(len(y_train_smote)),
        "original_class_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "train_class_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "smote_class_distribution": {
            str(k): int(v)
            for k, v in pd.Series(y_train_smote).value_counts().sort_index().items()
        },
        "feature_count": int(len(x.columns)),
        "feature_columns": x.columns.tolist(),
        "engineered_features": ENGINEERED_FEATURES.copy(),
        "model_config": XGB_CONFIG.copy(),
        "feature_importance": {
            name: float(value)
            for name, value in zip(x.columns, model.feature_importances_)
        },
    }

    bundle = LoanModelBundle(
        model=model,
        scaler=scaler,
        feature_columns=x.columns.tolist(),
        category_encoders=category_encoders,
    )
    return bundle, metrics


def _prepare_for_inference(
    bundle: LoanModelBundle, data: pd.DataFrame
) -> pd.DataFrame:
    encoded, _ = _clean_and_encode(data, bundle.category_encoders)
    return encoded.reindex(columns=bundle.feature_columns)


def predict_one(bundle: LoanModelBundle, applicant: dict[str, Any]) -> dict[str, Any]:
    encoded = _prepare_for_inference(bundle, pd.DataFrame([applicant]))
    if encoded.empty:
        raise ValueError("Applicant was removed by the age/employment sanity filters.")

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
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    source = data.copy()
    encoded = _prepare_for_inference(bundle, source)

    valid_source = source.loc[encoded.index]
    row_ids = (
        valid_source["loan_id"]
        if "loan_id" in valid_source.columns
        else pd.Series(
            range(1, len(valid_source) + 1),
            index=valid_source.index,
            name="loan_id",
        )
    )

    scaled = bundle.scaler.transform(encoded)
    predictions = bundle.model.predict(scaled).astype(int)
    probabilities = bundle.model.predict_proba(scaled)

    result = pd.DataFrame(
        {
            "loan_id": row_ids.values,
            "prediction": predictions,
            "predicted_status": [
                "Approved" if value == 1 else "Rejected"
                for value in predictions
            ],
            "approval_probability": probabilities[:, 1],
            "rejection_probability": probabilities[:, 0],
        },
        index=valid_source.index,
    )

    if "loan_status" in valid_source.columns:
        result["actual_status"] = valid_source["loan_status"].astype(int).values
        result["correct"] = result["prediction"] == result["actual_status"]

    return result.reset_index(drop=True)


def save_bundle(bundle: LoanModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> LoanModelBundle:
    return joblib.load(path)
