<div align="center">

# LOAN INTELLIGENCE

### NTI Machine Learning Track · Loan Status Classification

**An end-to-end machine learning project for predicting loan status from applicant data.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost-purple)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.47%25-success)
![App](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue?logo=githubactions&logoColor=white)

[Open in Google Colab](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)

</div>

---

## Overview

**Loan Intelligence** is an educational ML project built for the NTI Machine Learning track. It demonstrates the complete workflow from raw data to model evaluation and a simple deployed prediction demo.

The project focuses on **binary classification** of `loan_status` using applicant and loan features.

> **Note:** This is an educational/portfolio project, not a real credit-underwriting system. Model probabilities should not be interpreted as real-world lending decisions.

## Project Structure

```text
loan-prediction-nti/
│
├── app.py                         # Streamlit prediction demo
├── loan_data.csv                  # Dataset
├── Loan_Intelligence_Colab.ipynb  # Reproducible ML experiments
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python runtime for deployment
├── README.md
│
├── src/
│   ├── __init__.py
│   └── model.py                   # Training + preprocessing + inference
│
├── tests/
│   └── test_model.py              # Model/data tests
│
├── scripts/
│   └── full_model_benchmark.py    # Nine-model benchmark
│
└── docs/
    ├── architecture.md
    └── model-card.md
```

## ML Pipeline

```text
Raw Dataset
    ↓
Train / Test Split (80 / 20, stratified)
    ↓
IQR-based clipping + sanity filtering
    ↓
Categorical encoding
    ↓
RobustScaler
    ↓
SMOTETomek on training data only
    ↓
XGBoost
    ↓
Held-out Test Evaluation
```

The preprocessing decisions are fitted from the training split and reused during inference to reduce data leakage and keep the training/inference feature contract consistent.

## Model Benchmark

Nine classifiers were evaluated on the same held-out test split:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.42% | 0.6349 | 0.9155 | 0.7498 |
| KNN | 85.96% | 0.6292 | 0.8975 | 0.7397 |
| Decision Tree | 89.12% | 0.7148 | 0.8495 | 0.7763 |
| Random Forest | 91.64% | 0.7868 | 0.8560 | 0.8199 |
| Extra Trees | 91.28% | 0.7716 | 0.8630 | 0.8147 |
| Gradient Boosting | 90.84% | 0.7618 | 0.8555 | 0.8059 |
| HistGradientBoosting | 93.23% | 0.8866 | 0.7975 | 0.8397 |
| SVM | 88.43% | 0.6809 | 0.9025 | 0.7762 |
| **XGBoost** | **93.47%** | **0.8673** | **0.8335** | **0.8501** |

XGBoost was selected because it produced the strongest measured **accuracy and F1** in the benchmark.

### XGBoost Configuration

```text
n_estimators      = 900
max_depth         = 5
learning_rate     = 0.035
subsample         = 0.95
colsample_bytree  = 0.90
min_child_weight  = 1
gamma             = 0.0
reg_alpha         = 0.02
reg_lambda        = 1.0
random_state      = 42
eval_metric       = logloss
tree_method       = hist
n_jobs            = 2
```

### Confusion Matrix

```text
                 Predicted
                 0       1
Actual  0      6743    255
        1       333   1667
```

## Dataset

The repository contains approximately **45,000 rows** and the following model inputs:

```text
person_age
person_income
person_home_ownership
person_emp_exp
loan_intent
loan_amnt
loan_int_rate
loan_percent_income
cb_person_cred_hist_length
credit_score
previous_loan_defaults_on_file
person_gender
person_education
```

Target:

```text
loan_status
```

## Streamlit Demo

The application is intentionally simple: it is only for **testing the trained model**.

```text
Applicant Information
        ↓
Predict
        ↓
Approved / Rejected
        ↓
Approval + Rejection probabilities
        ↓
Model metrics
```

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first prediction trains the model from `loan_data.csv`; Streamlit then caches the trained model for subsequent predictions.

## Reproducible Notebook

`Loan_Intelligence_Colab.ipynb` provides a browser-based workflow for reproducing the ML experiment, including preprocessing, class balancing, model comparison and evaluation.

[Open the notebook in Google Colab](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)

## Testing

Run the test suite with:

```bash
pytest -q
```

GitHub Actions also runs the project's automated checks and the nine-model benchmark.

## Engineering Principles

- **Reproducible:** fixed random states and documented configuration.
- **Leakage-aware:** preprocessing decisions are fitted on training data only.
- **Modular:** reusable ML logic lives in `src/model.py`.
- **Tested:** core data/model behavior has automated tests.
- **Simple deployment:** Streamlit provides a focused prediction interface.
- **Honest evaluation:** accuracy is reported alongside precision, recall, F1 and the confusion matrix.

## Limitations

- The dataset is historical and is not evidence of real-world lending performance.
- The benchmark is based on one fixed stratified train/test split.
- Model probabilities are predictive scores, not calibrated financial approval odds.
- The application is intended for education and demonstration, not production credit decisions.

---

<div align="center">

**NTI Machine Learning Project · Loan Intelligence**

</div>
