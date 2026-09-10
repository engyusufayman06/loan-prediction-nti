# Architecture

## System flow

```text
                    ┌──────────────────────┐
                    │   loan_data.csv      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cleaning + Filters   │
                    │ age <= 80             │
                    │ employment <= 50      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ 5 derived features   │
                    └──────────┬───────────┘
                               │
                         train / test
                               │
                               ▼
                             SMOTE
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
   Scale-required models                    Tree models
             │                                   │
       StandardScaler                         XGBoost
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
                       Held-out evaluation
                               │
                               ▼
                         Streamlit UI
```

## Responsibilities

### `Loan_Intelligence_Colab.ipynb`

Final research notebook containing data integrity checks, statistical inference, feature engineering, nine-model benchmarking, ROC/PR evaluation, learning-curve assessment, global feature importance, XGBoost hyperparameter search, SHAP analysis, and artifact serialization.

### `src/model.py`

Reusable training and inference code. It keeps the final notebook's feature-engineering, encoding, SMOTE, and XGBoost sequence in one place so the application does not duplicate notebook logic. The stored scaler is retained as a notebook artifact; the deployable XGBoost path follows the notebook's tree-model rule and does not scale the tree features before inference.

### `app.py`

Presentation layer. It collects applicant inputs, generates the engineered features through the shared ML core, and exposes single-applicant, batch, and model-insight workflows. Streamlit caching prevents retraining on every interaction in the same session.

### `tests/`

Lightweight tests for the dataset contract, feature engineering, numeric encoding, and inference validation.

### `.github/workflows/ci.yml`

Runs compilation, tests, and the repository model-contract smoke check on pushes and pull requests targeting `main`.

## Design decisions

- **Feature engineering:** the final notebook adds five ratio/indicator features before model training.
- **SMOTE:** balances the training split only; the held-out test split remains untouched.
- **StandardScaler:** used by the notebook for scale-sensitive models. XGBoost, like the other tree-based benchmark models, is trained on the unscaled engineered representation.
- **XGBoost:** retained as the deployable repository model because the application's inference and explainability layer are built around it.
- **Benchmark transparency:** HistGradientBoosting is explicitly documented as the final notebook's benchmark leader rather than being hidden.
- **Streamlit:** used for a fast, presentation-ready ML demo without introducing an unnecessary backend layer for the NTI project.
