# Loan Intelligence — NTI Machine Learning Project

<p align="center">
  <strong>End-to-end Machine Learning system for loan approval prediction</strong><br>
  Exploratory analysis • preprocessing • class balancing • model benchmarking • interactive inference
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="ML" src="https://img.shields.io/badge/Track-Machine%20Learning-0EA5E9">
  <img alt="Best model" src="https://img.shields.io/badge/Best%20benchmark-XGBoost-7C3AED">
  <img alt="Accuracy" src="https://img.shields.io/badge/Test%20accuracy-92.87%25-22C55E">
</p>

## Overview

**Loan Intelligence** is an NTI Machine Learning project that studies whether a loan application is likely to be approved or rejected from applicant, credit-history, and loan characteristics.

The project follows a complete ML workflow:

```text
Raw data
   ↓
Data quality checks
   ↓
Outlier handling + filtering
   ↓
Categorical encoding
   ↓
Train / test split
   ↓
Robust scaling
   ↓
SMOTETomek on training data
   ↓
Model benchmarking
   ↓
XGBoost selection
   ↓
Interactive prediction app
```

> **Important:** This is a learning / portfolio project, not a production credit-underwriting system. A prediction should never be treated as the sole basis for a real financial decision.

## Project Results

The notebook benchmarked six classification models on the same held-out test set:

| Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.55% | 0.64 | **0.92** | 0.75 |
| KNN | 86.29% | 0.64 | 0.88 | 0.74 |
| Decision Tree | 89.28% | 0.73 | 0.82 | 0.77 |
| Random Forest | 89.48% | 0.71 | 0.88 | 0.79 |
| SVM | 88.02% | 0.67 | **0.92** | 0.77 |
| **XGBoost** | **92.87%** | **0.87** | 0.80 | **0.83** |

XGBoost is the selected benchmark winner because it achieved the strongest overall test accuracy, class-1 precision, and class-1 F1 in this comparison. Logistic Regression and SVM achieved higher class-1 recall, so the selection is a **trade-off**, not a claim that XGBoost is universally superior.

## Dataset

The repository contains `loan_data.csv` used by the notebook.

- Rows after preprocessing: **44,990**
- Processed features: **20**
- Target: `loan_status`
- Original training split before resampling: **27,991 class-0 / 8,001 class-1**
- After SMOTETomek on the training data: **27,887 / 27,887**

The notebook identifies the dataset source as **Kaggle**. The exact Kaggle URL is not recorded in the project files, so this README intentionally does not invent one.

## Features

The project works with applicant, loan, and credit-history information, including:

- Age and employment experience
- Annual income
- Home ownership
- Education
- Loan amount
- Loan interest rate
- Loan-to-income ratio
- Loan intent
- Credit score
- Credit-history length
- Previous loan defaults

One-hot encoding is used for nominal categorical variables. Education is ordinal-encoded in the original notebook.

## Preprocessing

The implemented workflow follows the notebook:

1. Remove `loan_id` because it is an identifier.
2. Clip selected numeric variables using IQR bounds.
3. Filter unrealistic values for age and employment experience.
4. Encode categorical variables.
5. Split data with `random_state=42`.
6. Fit `RobustScaler` **only on the training split**.
7. Apply `SMOTETomek` **only to the scaled training data**.
8. Train and evaluate multiple classifiers.

This ordering avoids fitting the scaler or resampler on the held-out test set.

## Interactive App

The repository includes a Streamlit interface for demonstrating inference.

### Run locally

```bash
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti

python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app trains the same XGBoost workflow from `loan_data.csv` and caches the trained model for the current session.

## Repository Structure

```text
loan-prediction-nti/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   └── model.py
├── tests/
│   └── test_model.py
├── app.py
├── loan.ipynb
├── loan_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook remains the **research / experimentation record**, reusable ML logic lives in `src/`, and Streamlit is the **presentation layer**.

## Model Card — Short Version

**Intended use:** educational demonstration of binary loan-status classification.

**Not intended for:** automated real-world lending decisions, credit scoring, regulatory decisions, or decisions affecting an applicant without human review.

**Known limitations:**
- The dataset may not represent a real lender's current applicant population.
- Correlation is not a complete measure of feature usefulness.
- Class balancing changes the training distribution.
- The benchmark has not been audited for fairness, calibration, or dataset shift.
- No real-time data integration is included.

## Next Steps

A stronger production-oriented version could add:

- FastAPI inference service
- Serialized model + versioned preprocessing artifact
- SHAP-based local explanations
- Probability calibration
- Fairness / subgroup evaluation
- Automated tests and CI/CD
- Docker deployment
- Experiment tracking with MLflow
- Model monitoring and drift detection

## Team

**NTI Machine Learning Track**

Team members:
1. ____________________
2. ____________________
3. ____________________
4. ____________________

## License

This repository is intended as an educational NTI project. Add the appropriate license before redistributing the dataset or code outside the course context.
