from pathlib import Path

import pandas as pd
import pytest

from src.model import MODEL_INPUT_COLUMNS, NUMERIC_CLIP_COLUMNS, _clean_and_encode, predict_batch


DATA_PATH = Path(__file__).resolve().parents[1] / "loan_data.csv"


def test_dataset_has_expected_target():
    df = pd.read_csv(DATA_PATH, nrows=10)
    assert "loan_status" in df.columns


def test_cleaning_removes_identifier_and_builds_iqr_bounds():
    df = pd.read_csv(DATA_PATH, nrows=100)
    X = df.drop(columns=["loan_status"])
    cleaned, bounds = _clean_and_encode(X)
    assert "loan_id" not in cleaned.columns
    assert bounds
    assert all(column in bounds for column in NUMERIC_CLIP_COLUMNS if column in X.columns)


def test_cleaning_returns_numeric_features():
    df = pd.read_csv(DATA_PATH, nrows=100)
    X = df.drop(columns=["loan_status"])
    cleaned, _ = _clean_and_encode(X)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in cleaned.dtypes)


def test_batch_inference_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch(None, pd.DataFrame({MODEL_INPUT_COLUMNS[0]: [30]}))
