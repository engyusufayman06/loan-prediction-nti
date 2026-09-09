from pathlib import Path

import pandas as pd

from src.model import NUMERIC_CLIP_COLUMNS, _clean_and_encode


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
    assert all(
        column in bounds for column in NUMERIC_CLIP_COLUMNS if column in X.columns
    )


def test_cleaning_returns_numeric_features():
    df = pd.read_csv(DATA_PATH, nrows=100)
    X = df.drop(columns=["loan_status"])
    cleaned, _ = _clean_and_encode(X)

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in cleaned.dtypes)
