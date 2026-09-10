# LOAN INTELLIGENCE

### NTI Machine Learning Track · End-to-End Loan Status Classification

**Raw applicant data → preprocessing → 9-model benchmark → XGBoost → Streamlit prediction app**

![Track](https://img.shields.io/badge/Track-Machine%20Learning-0EA5E9) ![Model](https://img.shields.io/badge/Model-XGBoost-7C3AED) ![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.47%25-22C55E) ![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)

---

## Project Overview

Loan Intelligence is an educational machine-learning project for **binary loan-status classification**. It packages the complete workflow into reusable training/inference code, model benchmarking, an interactive Streamlit application, tests, CI, and documentation.

> **Educational project:** this is not a production credit-underwriting system and should not be used for real lending decisions.

## ML Task

**Supervised binary classification** using `loan_status` as the target.

The model learns from applicant and loan characteristics instead of relying on a single hand-written decision rule.

## Dataset

The repository contains approximately **45,000 loan records** with fields including:

```text
person_age
person_gender
person_education
person_income
person_emp_exp
person_home_ownership
loan_amnt
loan_intent
loan_int_rate
loan_percent_income
cb_person_cred_hist_length
credit_score
previous_loan_defaults_on_file
loan_status
```

The model consumes 13 applicant/loan inputs. `loan_id`, when present, is treated as an identifier and ignored.

---

## ML Pipeline

```text
loan_data.csv
     ↓
Stratified 80/20 train-test split
     ↓
Training-only IQR bounds + sanity filtering
     ↓
Binary / ordinal / one-hot encoding
     ↓
RobustScaler
     ↓
SMOTETomek on training data only
     ↓
Model training
     ↓
Held-out test evaluation
```

The preprocessing is leakage-aware: data-dependent clipping bounds are fitted on the training split and then applied to the test split. The test set is never resampled.

---

## Model Benchmark

Nine classifiers were evaluated on the same held-out test split.

| Rank | Model | Accuracy | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|
| 1 | **XGBoost** | **93.47%** | 0.8673 | 0.8335 | **0.8501** |
| 2 | HistGradientBoosting | 93.23% | 0.8866 | 0.7975 | 0.8397 |
| 3 | Random Forest | 91.64% | 0.7868 | 0.8560 | 0.8199 |
| 4 | Extra Trees | 91.28% | 0.7716 | 0.8630 | 0.8147 |
| 5 | Gradient Boosting | 90.84% | 0.7618 | 0.8555 | 0.8059 |
| 6 | Decision Tree | 89.12% | 0.7148 | 0.8495 | 0.7763 |
| 7 | SVM | 88.43% | 0.6809 | 0.9025 | 0.7762 |
| 8 | Logistic Regression | 86.42% | 0.6349 | 0.9155 | 0.7498 |
| 9 | KNN | 85.96% | 0.6292 | 0.8975 | 0.7397 |

### Why XGBoost?

XGBoost produced the strongest measured test accuracy and F1 score in the benchmark, so it was selected as the main application model.

The comparison also shows the precision/recall trade-off: Logistic Regression and SVM achieve higher class-1 recall, while XGBoost gives the strongest overall balance in this benchmark.

**Best measured test accuracy: 93.47%.**

---

## XGBoost Configuration

The current reusable model in `src/model.py` uses:

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

The configuration uses a low learning rate, subsampling, column subsampling and regularization to control complexity while keeping strong predictive performance.

---

## Class Imbalance

The training pipeline uses **SMOTETomek** to address the imbalanced target distribution.

```text
Training data → SMOTETomek → more balanced training distribution
```

Resampling is performed only on the training set. Evaluation remains on untouched held-out data.

---

## XGBoost Test Behavior

For the 93.47% benchmark run:

```text
                    Predicted
                 0           1
Actual  0      6743        255
        1       333       1667
```

Class-1 metrics:

- **Precision:** 0.8673
- **Recall:** 0.8335
- **F1:** 0.8501

---

## Repository Structure

```text
loan-prediction-nti/
├── app.py                         # Streamlit application
├── src/model.py                   # Reusable ML core
├── scripts/full_model_benchmark.py# 9-model benchmark
├── tests/                         # Automated tests
├── docs/                          # Documentation
├── assets/                        # UI assets
├── loan_data.csv                  # Dataset
├── Loan_Intelligence_Colab.ipynb  # Colab reproduction
└── .github/workflows/             # CI / deployment workflows
```

### Engineering principle

```text
Notebook  = research / experimentation
src/      = reusable ML logic
app.py    = product + presentation layer
tests/    = quality contract
docs/     = technical documentation
```

---

## Application Features

### Prediction Studio

- complete applicant input form;
- real model inference;
- predicted class and probability;
- prediction history;
- benchmark context.

### Batch Lab

```text
Upload CSV
   ↓
Schema validation
   ↓
Same preprocessing contract
   ↓
Batch predictions
   ↓
Probabilities + labels
   ↓
Optional evaluation
   ↓
Download results
```

### Model Insights

- model comparison;
- benchmark metrics;
- confusion matrix;
- feature importance;
- precision/recall trade-offs;
- model limitations.

---

## Reproduction

### Install

```bash
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the 9-model benchmark

```bash
python scripts/full_model_benchmark.py
```

### Google Colab

[Open Loan Intelligence in Google Colab](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)

---

## Quality & CI

The repository uses automated tests and GitHub Actions to catch regressions. The benchmark script is also stored in the repository so model comparisons can be reproduced instead of relying only on manually written README numbers.

---

## Limitations

- Historical educational dataset.
- Accuracy alone does not measure fairness or lending suitability.
- The model is not validated for real credit decisions.
- Threshold choice changes precision/recall trade-offs.
- No claim of production readiness is made.

---

## Roadmap

- probability calibration and threshold analysis;
- stronger cross-validation;
- SHAP explainability;
- model monitoring and drift checks;
- containerized deployment;
- systematic hyperparameter search.

---

Built for the **NTI Machine Learning Track** for educational and portfolio purposes.
