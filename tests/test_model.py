from pathlib import Path

import pandas as pd
import pytest

from src.model import (
    ENGINEERED_FEATURES,
    MODEL_INPUT_COLUMNS,
    _clean_and_encode,
    predict_batch,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "loan_data.csv"


def test_dataset_has_expected_shape_and_target():
    df = pd.read_csv(DATA_PATH, nrows=10)
    assert df.shape[1] == 14
    assert "loan_status" in df.columns


def test_final_feature_engineering_is_numeric():
    df = pd.read_csv(DATA_PATH, nrows=100)
    X = df.drop(columns=["loan_status"])
    cleaned, encoders = _clean_and_encode(X)

    assert set(ENGINEERED_FEATURES).issubset(cleaned.columns)
    assert len(cleaned.columns) == 18
    assert encoders
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in cleaned.dtypes)


def test_cleaning_applies_final_notebook_filters():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["loan_status"])
    cleaned, _ = _clean_and_encode(X)

    assert cleaned["person_age"].max() <= 80
    assert cleaned["person_emp_exp"].max() <= 50
    assert len(cleaned) == 44988


def test_model_input_contract_has_13_raw_features():
    assert len(MODEL_INPUT_COLUMNS) == 13


def test_batch_inference_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch(None, pd.DataFrame({MODEL_INPUT_COLUMNS[0]: [30]}))
