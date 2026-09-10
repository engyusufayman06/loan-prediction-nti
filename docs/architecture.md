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
                    │ age <= 80            │
                    │ employment <= 50     │
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
       StandardScaler                  HistGradientBoosting
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

Final research notebook containing data integrity checks, statistical inference, feature engineering, nine-model benchmarking, ROC/PR evaluation, learning-curve assessment, global feature importance, the separate XGBoost hyperparameter search, SHAP analysis, and artifact serialization.

### `src/model.py`

Reusable training and inference code. It keeps the final notebook's feature-engineering, encoding, SMOTE, and HistGradientBoosting sequence in one place so the application does not duplicate notebook logic. HistGradientBoosting follows the notebook's unscaled tree-model path.

### `app.py`

Presentation layer. It collects applicant inputs, generates the engineered features through the shared ML core, and exposes single-applicant, batch, and model-insight workflows. Streamlit caching prevents retraining on every interaction in the same session.

### `tests/`

Lightweight tests for the dataset contract, feature engineering, numeric encoding, and inference validation.

### `.github/workflows/ci.yml`

Runs compilation, tests, and the repository model-contract smoke check on pushes and pull requests targeting `main`.

## Design decisions

- **Feature engineering:** the final notebook adds five ratio/indicator features before model training.
- **SMOTE:** balances the training split only; the held-out test split remains untouched.
- **StandardScaler:** used by the notebook for scale-sensitive models. HistGradientBoosting, like the other tree-based benchmark models, is trained on the unscaled engineered representation.
- **HistGradientBoosting:** selected as the deployable repository model because it is the final notebook benchmark leader with **93.2874% accuracy**, **0.846545 F1**, and **0.976527 ROC-AUC**.
- **Feature attribution:** HistGradientBoosting has no native `feature_importances_`, so the application presents permutation-based relative importance from the notebook's cross-model analysis.
- **Benchmark transparency:** all nine classifiers remain visible so the selection is reproducible rather than hidden behind a single algorithm.
- **Streamlit:** used for a fast, presentation-ready ML demo without introducing an unnecessary backend layer for the NTI project.
