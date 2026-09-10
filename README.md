<div align="center">

# ◈ LOAN INTELLIGENCE
### NTI Machine Learning Track · End-to-End Loan Status Classification

**From raw applicant data → reproducible ML pipeline → engineered features → benchmarked models → interactive decision studio → deployable demo.**

<br>

<img src="https://img.shields.io/badge/Track-Machine%20Learning-06B6D4?style=for-the-badge">
<img src="https://img.shields.io/badge/Model-XGBoost-7C3AED?style=for-the-badge">
<img src="https://img.shields.io/badge/Benchmark%20Leader-HistGradientBoosting-22C55E?style=for-the-badge">
<img src="https://img.shields.io/badge/XGBoost%20Test%20Accuracy-92.97%25-22C55E?style=for-the-badge">
<img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">

<br><br>

<a href="https://github.com/engyusufayman06/loan-prediction-nti">Repository</a> ·
<a href="https://www.nti.sci.eg/">NTI</a>

</div>

---

## ⚡ The project in 30 seconds

**Loan Intelligence** is an educational machine-learning system for **binary loan-status classification**. Instead of stopping at a notebook, the project packages the experiment into a small product: a reusable preprocessing/model layer, an interactive Streamlit application, batch CSV inference, model diagnostics, feature engineering, benchmarking, and an in-app technical presentation.

> **Final notebook benchmark:** HistGradientBoosting reached **93.2874% held-out accuracy** and **0.846545 F1**. The repository's deployable XGBoost model reached **92.9651% accuracy**, **0.855434 class-1 precision**, **0.8225 class-1 recall**, **0.838644 F1**, and **0.975957 ROC-AUC** on the same held-out test split.

This repository is intentionally transparent: metrics, preprocessing decisions, engineered features, class balancing, model trade-offs, limitations, and the separate hyperparameter-search experiment are documented rather than hidden behind a single accuracy number.

---

## 🎬 Experience the system

The Streamlit app is organized as a product workspace:

| Workspace | What it does |
|---|---|
| **Prediction Studio** | Enter a complete applicant profile and run XGBoost inference. |
| **Batch Lab** | Upload a CSV, validate its schema, score rows, inspect results, and download predictions. |
| **Model Insights** | Explore the complete nine-model benchmark, ROC-AUC, confusion matrix, XGBoost feature importance, dataset statistics, and optimization results. |
| **Team & About** | Project identity, team members, and NTI context. |

### Product flow

```text
                    ┌─────────────────────────┐
                    │      loan_data.csv       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Research / EDA Notebook │
                    └────────────┬────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │       Feature Engineering      │
                 │  ratios → risk interaction     │
                 │       → rental indicator       │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │      Reusable ML Core          │
                 │ cleaning → encoding → SMOTE   │
                 │      → StandardScaler → XGB    │
                 └───────────────┬───────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
       ┌────────────────┐ ┌───────────────┐ ┌───────────────┐
       │ Prediction     │ │ Batch Lab     │ │ Model Insights│
       │ Studio         │ │ CSV scoring   │ │ Diagnostics   │
       └────────────────┘ └───────────────┘ └───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Interactive ML Demo     │
                    └─────────────────────────┘
```

---

# 01 · Problem framing

### Objective

Build a **binary classification** model that predicts `loan_status` from applicant and loan characteristics.

### Why this is an ML problem

The system learns a mapping from historical labelled examples to a binary target. The project therefore focuses on:

- supervised learning;
- preprocessing and feature representation;
- engineered domain-style features;
- class-imbalance handling;
- model comparison;
- held-out evaluation;
- reusable inference.

### Important boundary

This is **not** a production credit-underwriting engine. It is an NTI educational / portfolio project designed to demonstrate an end-to-end ML workflow.

---

# 02 · Dataset snapshot

| Property | Value |
|---|---:|
| Source file | `loan_data.csv` |
| Raw rows | **45,000** |
| Raw columns | **14** |
| Rows after final cleaning | **44,988** |
| Model features | **18** |
| Train rows | **35,990** |
| Test rows | **8,998** |
| Training rows after SMOTE | **55,980** |
| Original class 0 | **35,000** |
| Original class 1 | **10,000** |
| Original class-1 share | **22.22%** |
| Training class 0 before SMOTE | **27,990** |
| Training class 1 before SMOTE | **8,000** |
| Balanced class 0 after SMOTE | **27,990** |
| Balanced class 1 after SMOTE | **27,990** |

The final notebook removes rows with `person_age > 80` or `person_emp_exp > 50`, then engineers five additional features before encoding and splitting.

### Raw inference schema

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

`loan_id` may be included in batch files as an identifier and is ignored by the model. `loan_status` may be included in labelled uploads for evaluation; it is **not** used as an inference feature.

---

# 03 · ML pipeline

The final notebook uses the following sequence:

```text
Raw data
   │
   ├── remove loan_id when present
   │
   ├── age / employment sanity filters
   │      age <= 80
   │      employment experience <= 50
   │
   ├── engineer 5 domain-style features
   │
   ├── LabelEncoder on object columns
   │
   ├── 80/20 stratified train-test split
   │      random_state=42
   │
   ├── SMOTE on training split only
   │
   ├── StandardScaler
   │
   └── model benchmarking
   │
   ▼
XGBoost deployable inference model
```

### Engineered features

| Feature | Formula / logic |
|---|---|
| `income_to_loan_ratio` | `person_income / (loan_amnt + 1)` |
| `emp_length_to_age_ratio` | `person_emp_exp / (person_age + 1)` |
| `cred_hist_to_age_ratio` | `cb_person_cred_hist_length / (person_age + 1)` |
| `high_risk_income_ratio` | `loan_percent_income * loan_int_rate` |
| `is_renting` | `1` when home ownership is `RENT`, otherwise `0` |

These features increase the final model representation from the raw applicant contract to **18 model features**.

### Why the pipeline matters

The important engineering step is not simply training XGBoost. The same feature contract and transformations are reused during inference so that a prediction in the UI follows the project's trained representation rather than a separate, hand-written preprocessing path.

---

# 04 · Model benchmark

The final notebook evaluates **nine classifiers** on the same stratified held-out test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **HistGradientBoosting** | **93.2874%** | 0.860537 | 0.8330 | **0.846545** | **0.976527** |
| **XGBoost** | **92.9651%** | **0.855434** | 0.8225 | **0.838644** | 0.975957 |
| Random Forest | 91.1202% | 0.773079 | **0.8500** | 0.809717 | 0.968021 |
| Extra Trees | 89.7755% | 0.738305 | 0.8365 | 0.784341 | 0.963371 |
| Gradient Boosting | 89.3532% | 0.717809 | 0.8585 | 0.781876 | 0.962550 |
| SVM | 87.7639% | 0.671630 | 0.8795 | 0.761637 | 0.950739 |
| Decision Tree | 88.0862% | 0.703152 | 0.8030 | 0.749767 | 0.853058 |
| Logistic Regression | 86.1747% | 0.635971 | 0.8840 | 0.739749 | 0.944234 |
| KNN | 86.2192% | 0.643396 | 0.8525 | 0.733333 | 0.926571 |

### Selection logic

The final notebook sorts the benchmark by **F1-Score**, making **HistGradientBoosting** the benchmark leader:

- Accuracy: **93.2874%**
- Precision: **0.860537**
- Recall: **0.8330**
- F1: **0.846545**
- ROC-AUC: **0.976527**

The repository keeps **XGBoost as the deployable model** because the application, feature-importance presentation, and inference layer are built around the XGBoost contract.

XGBoost itself reports:

- Accuracy: **92.9651%**
- Precision: **0.855434**
- Recall: **0.8225**
- F1: **0.838644**
- ROC-AUC: **0.975957**

That distinction is deliberate: the README does not pretend that XGBoost won every metric when the final notebook shows HistGradientBoosting slightly ahead.

---

# 05 · What happens to class imbalance?

Before resampling, the final notebook has:

```text
Full dataset
Class 0  35,000
Class 1  10,000

Training split
Class 0  27,990
Class 1   8,000
```

After SMOTE:

```text
Class 0  27,990
Class 1  27,990
Total    55,980
```

SMOTE is applied **only to the training split**. The held-out test set remains untouched for evaluation.

---

# 06 · Model behavior, not just accuracy

### XGBoost confusion matrix

```text
                    Predicted
                 0           1
Actual  0      6720        278
        1       355       1645
```

From this matrix:

- True Negatives: **6,720**
- False Positives: **278**
- False Negatives: **355**
- True Positives: **1,645**
- Error rate: **7.0349%**
- Correct prediction rate: **92.9651%**
- False Positive Rate: **3.9769%**
- False Negative Rate: **17.7500%**

For class 1, the final notebook reports:

- **Precision:** 0.855434
- **Recall:** 0.8225
- **F1:** 0.838644

The right operating point depends on the business cost of false positives versus false negatives.

---

# 07 · Generalization & overfitting

The final notebook includes a **learning-curve assessment** using three-fold stratified cross-validation and F1 as the scoring metric.

The benchmark is therefore interpreted using held-out performance and cross-validation diagnostics rather than training accuracy alone.

The project does **not** claim that the model is perfect or production ready.

Known facts from the benchmark:

- HistGradientBoosting: **93.2874%** held-out accuracy.
- XGBoost: **92.9651%** held-out accuracy.
- Logistic Regression: **86.1747%** held-out accuracy.
- The difference between models demonstrates why the project includes a benchmark instead of presenting a single algorithm in isolation.

---

# 08 · Feature representation

The final notebook transforms the raw schema into an **18-feature** model representation.

It uses:

- LabelEncoder for object/categorical columns;
- five engineered ratio/indicator features;
- a stratified 80/20 split;
- SMOTE for the training split;
- StandardScaler after resampling.

### Top XGBoost features

The notebook's global feature-importance chart shows the following top XGBoost features, rounded for presentation:

| Rank | Feature | Relative importance |
|---:|---|---:|
| 1 | `previous_loan_defaults_on_file` | **~88.0%** |
| 2 | `high_risk_income_ratio` | **~1.9%** |
| 3 | `cb_person_cred_hist_length` | **~1.4%** |
| 4 | `loan_int_rate` | **~1.0%** |
| 5 | `person_home_ownership` | **~0.8%** |
| 6 | `person_education` | **~0.7%** |
| 7 | `loan_intent` | **~0.7%** |
| 8 | `person_gender` | **~0.6%** |

> **Important:** model feature importance is a measure of how the fitted model uses a feature. It is not a causal claim that the feature alone determines a loan decision.

The notebook also performs global feature-importance cross-examination across the nine models, using native importance when available and permutation importance where required.

---

# 09 · Application architecture

```text
┌──────────────────────────────────────────────────────┐
│                 Streamlit Application                │
│                                                      │
│  Prediction Studio │ Batch Lab │ Model Insights     │
└──────────────────────────────┬───────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────┐
│                    src/model.py                      │
│                                                      │
│ train_model()  →  LoanModelBundle                   │
│ predict_one()  →  single applicant inference        │
│ predict_batch()→  CSV / dataframe inference         │
└──────────────────────────────┬───────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────┐
│ feature engineering + LabelEncoder + SMOTE +        │
│ StandardScaler + XGBoost                             │
└──────────────────────────────────────────────────────┘
```

### Design principle

**Notebook = research.**  
**`src/` = reusable ML logic.**  
**`app.py` = product / presentation layer.**  
**`tests/` = quality contract.**  
**`docs/` = engineering and model documentation.**

---

# 10 · Repository map

```text
loan-prediction-nti/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── full-model-benchmark.yml
│
├── .streamlit/
│   └── config.toml
│
├── docs/
│   ├── architecture.md
│   └── model-card.md
│
├── src/
│   ├── __init__.py
│   └── model.py
│
├── tests/
│   └── test_model.py
│
├── scripts/
│   └── full_model_benchmark.py
│
├── app.py
├── Loan_Intelligence_Colab.ipynb
├── demo_test.csv
├── loan_data.csv
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

# 11 · Interactive features

### Prediction Studio

Enter the complete 13-field applicant input contract. The application automatically creates the five engineered features used by the final notebook and sends the row through the reusable preprocessing/model layer.

### Batch Lab

```text
CSV template
     ↓
Upload
     ↓
Schema validation
     ↓
Feature engineering
     ↓
Batch inference
     ↓
Predictions + probabilities
     ↓
Optional labelled-data evaluation
     ↓
Download scored CSV
```

### Model Insights

The application surfaces:

- all nine benchmark models;
- Accuracy, Precision, Recall, F1 and ROC-AUC;
- dataset and split statistics;
- SMOTE balancing statistics;
- XGBoost confusion matrix;
- false-positive / false-negative rates;
- top XGBoost features;
- the notebook's separate hyperparameter-search result.

### Demo CSV

`demo_test.csv` contains six valid applicant records covering different education, income, home ownership, loan intent, credit-score, and previous-default scenarios.

---

# 12 · Run it locally

### Windows — recommended

```powershell
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is blocked:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Run tests:

```bash
python -m pytest -q
```

Run the benchmark script:

```bash
python scripts/full_model_benchmark.py
```

---

# 13 · Deploy

The repository is structured for **Streamlit Community Cloud**.

Use:

```text
Repository: engyusufayman06/loan-prediction-nti
Branch:     main
Main file:  app.py
```

The application reads `loan_data.csv` from the repository root and caches the trained model during normal Streamlit interaction.

---

# 14 · Engineering quality

This project goes beyond a notebook by including:

- reusable model/inference functions;
- a typed `LoanModelBundle` abstraction;
- single-row and batch inference paths;
- input/schema validation;
- deterministic feature engineering;
- class balancing on the training split only;
- unit tests;
- GitHub Actions CI;
- dedicated architecture documentation;
- a model card;
- an explicit benchmark;
- a demo CSV;
- explicit limitations and responsible-use guidance.

---

# 15 · Limitations & responsible use

This system is an **educational / portfolio ML demonstration**.

It has not been validated as a real lender's underwriting model and should not be used as the sole basis for a real financial decision.

Known limitations include:

- the dataset may not represent a real lender's current population;
- categorical LabelEncoder values are learned from the supplied dataset;
- no fairness audit or subgroup performance analysis has been completed;
- probability calibration has not been established for real-world approval odds;
- dataset shift and production drift are not monitored;
- the benchmark does not establish causal relationships;
- SMOTE changes the training distribution;
- the final notebook's exploratory preprocessing and benchmark are educational rather than a regulated credit-risk methodology;
- the top-feature chart is model attribution, not causal inference;
- the separate hyperparameter-search experiment is scored by F1 and is not automatically promoted to the deployed model.

---

# 16 · Production roadmap

```text
CURRENT
  │
  ├── Interactive Streamlit application
  ├── Reusable ML core
  ├── Batch inference
  ├── Engineered features
  └── Automated tests / CI
  │
  ▼
NEXT
  │
  ├── Versioned model artifact
  ├── FastAPI inference service
  ├── SHAP local explanations
  ├── Probability calibration
  ├── Fairness / subgroup evaluation
  └── Dockerized deployment
  │
  ▼
PRODUCTION MATURITY
  │
  ├── MLflow experiment tracking
  ├── Model registry
  ├── Data / concept drift monitoring
  ├── Observability + alerting
  └── Reproducible CI/CD
```

---

# 17 · Team

### NTI Machine Learning Track

| # | Team member |
|---:|---|
| 01 | **Yusuf Ayman Tolba** |
| 02 | **Mohamed Reda Hussein** |
| 03 | **Abdelrahman Mohamed Ahmed** |
| 04 | **Abdelmoniem Ibrahim Abdelmoniem** |
| 05 | **Asmaa Rabee Mohammed** |

---

# 18 · NTI

This project was developed in the context of the **National Telecommunication Institute (NTI) Machine Learning Track**.

**Official NTI:** https://www.nti.sci.eg/
