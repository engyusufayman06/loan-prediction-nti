# ◈ Loan Intelligence — NTI Machine Learning Project

<p align="center">
  <strong>End-to-end Machine Learning system for loan status prediction</strong><br>
  Data exploration • preprocessing • imbalance handling • model benchmarking • interactive inference
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Track" src="https://img.shields.io/badge/Track-Machine%20Learning-0EA5E9">
  <img alt="Model" src="https://img.shields.io/badge/Selected%20model-XGBoost-7C3AED">
  <img alt="Accuracy" src="https://img.shields.io/badge/Test%20accuracy-92.87%25-22C55E">
  <img alt="UI" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B">
</p>

> **Loan Intelligence** is an NTI Machine Learning portfolio project that turns a loan-status classification experiment into a reusable ML pipeline and a polished interactive product interface.

---

## 01 — Product overview

The project predicts whether a loan application is likely to be **approved (1)** or **rejected (0)** from applicant, financial, loan, and credit-history attributes.

The repository is intentionally structured as an end-to-end system rather than a single notebook:

```text
                 ┌──────────────────────┐
                 │     loan_data.csv    │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Data quality + EDA   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Clean + encode       │
                 │ IQR clipping         │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Train / test split   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ RobustScaler         │
                 │ SMOTETomek           │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Model benchmark      │
                 │ 6 classifiers        │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ XGBoost selected     │
                 └──────────┬───────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────┐                      ┌────────────────┐
│ Reusable ML   │                      │ Streamlit UI   │
│ src/model.py  │                      │ prediction +   │
│ tests         │                      │ presentation   │
└───────────────┘                      └────────────────┘
```

---

## 02 — Results

Six classification models were benchmarked on the same held-out test set:

| Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.55% | 0.64 | **0.92** | 0.75 |
| KNN | 86.29% | 0.64 | 0.88 | 0.74 |
| Decision Tree | 89.28% | 0.73 | 0.82 | 0.77 |
| Random Forest | 89.48% | 0.71 | 0.88 | 0.79 |
| SVM | 88.02% | 0.67 | **0.92** | 0.77 |
| **XGBoost** | **92.87%** | **0.87** | 0.80 | **0.83** |

### Selection logic

XGBoost is the selected benchmark model because it achieved the strongest combination of **test accuracy, class-1 precision, and class-1 F1** in this comparison.

That does **not** mean it is universally the best model. Logistic Regression and SVM achieved higher class-1 recall (0.92 vs 0.80), so the benchmark exposes an important precision/recall trade-off.

---

## 03 — Dataset & features

The repository contains the `loan_data.csv` dataset used by the notebook.

- **Rows after preprocessing:** 44,990
- **Processed model features:** 20
- **Target:** `loan_status`
- **Original training distribution:** 27,991 class-0 / 8,001 class-1
- **After SMOTETomek:** 27,887 / 27,887

Main feature groups:

- Applicant: age, gender, education, employment experience
- Financial: annual income
- Housing: home ownership
- Loan: amount, interest rate, loan-to-income ratio, intent
- Credit: credit score, credit-history length, previous loan defaults

The notebook identifies the dataset source as Kaggle, but the exact Kaggle URL is not recorded in the project files, so it is intentionally not fabricated here.

---

## 04 — Machine Learning pipeline

The implemented workflow follows the project notebook:

1. Remove `loan_id` because it is an identifier.
2. Clip selected numeric variables using IQR bounds.
3. Filter unrealistic `person_age` and `person_emp_exp` values.
4. Encode binary/ordinal variables and one-hot encode nominal categories.
5. Split into train/test with `random_state=42`.
6. Fit `RobustScaler` on the training split.
7. Apply `SMOTETomek` to the scaled training data only.
8. Train XGBoost.
9. Evaluate on the untouched test split.
10. Reuse the same preprocessing contract for single-applicant inference.

The reusable implementation lives in `src/model.py`; the notebook remains the experimentation record.

---

## 05 — Interactive product UI

The Streamlit app is more than a form. It contains four workspaces:

### Prediction

A production-inspired dark FinTech interface for entering an applicant profile and running the real XGBoost inference pipeline.

### Project Presentation

An **in-app presentation deck** with 9 slides covering:

1. The problem
2. The data
3. The ML pipeline
4. Model benchmark
5. Why XGBoost
6. The product/UI
7. Limitations
8. Production roadmap
9. Final takeaway

### Model Insights

Includes:

- Benchmark accuracy chart
- Full model comparison table
- XGBoost feature-importance view
- Precision/recall trade-off explanation
- Overfitting/generalization note

### Prediction History

Keeps the latest predictions in the current browser session so the user can review recent decisions without a database.

---

## 06 — Repository architecture

```text
loan-prediction-nti/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── docs/
│   ├── architecture.md
│   └── model-card.md
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

### Design principle

```text
Notebook       = research / experimentation
src/           = reusable ML logic
app.py         = product / presentation layer
tests/         = quality checks
.github/       = CI automation
docs/          = architecture + model governance notes
```

---

## 07 — Run locally

### Windows — simplest route

If your Python installation is managed by MSYS2, use your normal Windows Python executable rather than the MSYS2 Python environment.

Standard setup:

```bash
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501`.

The model is trained from `loan_data.csv` on first launch and cached for the current Streamlit session.

---

## 08 — Quality & engineering

The repository includes:

- Reusable training/inference functions
- A `LoanModelBundle` containing the model, scaler, feature schema, and clipping bounds
- Unit tests for core data assumptions
- GitHub Actions CI for syntax checks and tests
- Streamlit theme configuration
- Architecture and model-card documentation
- Clear separation between experimentation and application code

---

## 09 — Limitations & responsible use

This is an **educational / portfolio ML demonstration**, not a production credit-underwriting system.

Known limitations include:

- The dataset may not represent a real lender's current applicant population.
- Correlation is not a complete measure of feature usefulness.
- Class balancing changes the training distribution.
- The benchmark has not been audited for fairness, calibration, or dataset shift.
- No real-time data integration is included.
- Model probabilities should not be interpreted as guaranteed real-world approval odds.

**Never use this model as the sole basis for a real financial decision.**

---

## 10 — Production roadmap

The project can evolve toward a stronger production architecture with:

- FastAPI inference service
- Versioned serialized model + preprocessing artifact
- SHAP local explanations
- Probability calibration
- Fairness / subgroup evaluation
- Docker deployment
- MLflow experiment tracking
- Model monitoring and drift detection
- CI/CD deployment gates
- Authentication and audit logging

---

## 11 — Team

**NTI Machine Learning Track**

Team members:

1. ____________________
2. ____________________
3. ____________________
4. ____________________

---

## License

This repository is intended as an educational NTI project. Add the appropriate license before redistributing the dataset or code outside the course context.
