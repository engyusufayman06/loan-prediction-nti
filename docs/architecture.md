# Architecture

## System flow

```text
                    ┌──────────────────────┐
                    │   loan_data.csv      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cleaning + Encoding  │
                    │ IQR / filters / OHE  │
                    └──────────┬───────────┘
                               │
                         train / test
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
      RobustScaler                         held-out test
             │
             ▼
        SMOTETomek
             │
             ▼
          XGBoost
             │
             ▼
      Prediction + score
             │
             ▼
        Streamlit UI
```

## Responsibilities

### `loan.ipynb`
Research notebook containing EDA, preprocessing experiments, model benchmarking, confusion matrices, and the original project narrative.

### `src/model.py`
Reusable training and inference code. It keeps the preprocessing sequence in one place so the application does not duplicate notebook logic.

### `app.py`
Presentation layer. It collects applicant inputs and calls the reusable inference function. Streamlit caching prevents retraining on every interaction in the same session.

### `tests/`
Lightweight smoke tests for the dataset contract and preprocessing output.

### `.github/workflows/ci.yml`
Runs Python compilation and tests on pushes and pull requests targeting `main`.

## Design decisions

- **RobustScaler:** chosen in the notebook because several numeric variables contain outliers.
- **SMOTETomek:** combines minority oversampling with Tomek-link cleaning and is applied to the training split only.
- **XGBoost:** selected from the benchmark because it had the strongest overall held-out accuracy, class-1 precision, and class-1 F1.
- **Streamlit:** used for a fast, presentation-ready ML demo without introducing an unnecessary backend layer for the NTI project.
