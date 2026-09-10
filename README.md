<div align="center">

# LOAN INTELLIGENCE
### NTI Machine Learning Track · End-to-End Loan Status Classification

**From raw applicant data → reproducible ML pipeline → benchmarked models → interactive decision studio → deployable demo.**

<img src="https://img.shields.io/badge/Track-Machine%20Learning-0EA5E9">
<img src="https://img.shields.io/badge/Model-XGBoost-7C3AED">
<img src="https://img.shields.io/badge/Baseline%20Test%20Accuracy-93.47%25-22C55E">
<img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white">

[Repository](https://github.com/engyusufayman06/loan-prediction-nti) · [NTI](https://www.nti.sci.eg/) · [Colab Notebook](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)

</div>

---

## 01 — Project Overview

**Loan Intelligence** is an educational machine-learning system for **binary loan-status classification**. Instead of stopping at a notebook, the project packages the experiment into a small end-to-end ML product:

- reusable preprocessing, training and inference code;
- nine-model benchmark comparison;
- class-imbalance handling with SMOTETomek;
- a tuned XGBoost training configuration;
- an interactive Streamlit application;
- single-applicant prediction;
- batch CSV scoring and evaluation;
- model diagnostics and feature insights;
- an in-app technical presentation;
- reproducible Google Colab execution;
- unit tests and GitHub Actions CI;
- architecture, quickstart and model-card documentation.

> **Important:** this is an NTI educational / portfolio project. It is not a production credit-underwriting system, and its model probabilities must not be interpreted as real-world approval odds.

The repository is intentionally transparent: preprocessing decisions, benchmark trade-offs, class balancing, limitations and production considerations are documented rather than hidden behind a single accuracy number.

---

## 02 — The Project in One View

```text
loan_data.csv
      ↓
Research / EDA notebook
      ↓
Reusable ML core — src/model.py
      ↓
Cleaning → Encoding → Scaling → SMOTETomek
      ↓
Model benchmarking → XGBoost selection / tuning
      ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ Prediction Studio│    Batch Lab     │   Model Insights │
│ single applicant │ CSV inference    │ metrics / charts  │
└──────────────────┴──────────────────┴──────────────────┘
                         ↓
               Interactive presentation
```

### Product Workspaces

| Workspace | Purpose |
|---|---|
| **Prediction Studio** | Enter a complete applicant profile and run real model inference. |
| **Batch Lab** | Upload a CSV, validate its schema, score multiple applicants, evaluate labelled data, and download results. |
| **Model Insights** | Inspect the benchmark, confusion matrix, feature importance and precision/recall trade-offs. |
| **Project Presentation** | Follow the complete technical story through an interactive in-app presentation. |
| **Colab** | Reproduce the training and evaluation workflow directly in the browser. |

---

## 03 — Problem Framing

### Objective

Build a **binary classification** model that predicts `loan_status` from applicant and loan characteristics.

### Why this is an ML problem

The system learns a mapping from historical labelled examples to a binary target. The project therefore demonstrates:

- supervised learning;
- data cleaning and validation;
- feature representation;
- class-imbalance handling;
- model comparison;
- held-out evaluation;
- reusable inference;
- deployment-oriented engineering.

### Prediction target

`loan_status` is the binary target used during training and evaluation.

### Important boundary

This project demonstrates the engineering and modeling workflow around a loan-classification problem. It does **not** establish that the resulting model is suitable for real lending decisions.

---

## 04 — Dataset

The repository contains `loan_data.csv` with approximately **45,000 rows**. The original dataset schema contains the following fields:

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

### Dataset snapshot used by the original benchmark

| Property | Value |
|---|---:|
| Source file | `loan_data.csv` |
| Rows after preprocessing | **44,990** |
| Encoded model features | **20** |
| Target | `loan_status` |
| Original training class 0 | **27,991** |
| Original training class 1 | **8,001** |
| Post-SMOTETomek class 0 | **27,887** |
| Post-SMOTETomek class 1 | **27,887** |

The research notebook identifies the dataset source as Kaggle. The exact dataset URL is not recorded in the project files, so no unverified source link is fabricated here.

### Raw inference contract

The model consumes 13 applicant/loan fields:

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

`loan_id`, when present in a batch upload, is treated as an identifier and ignored by the model. `loan_status`, when present in an uploaded CSV, is treated as an evaluation label and is never used as an inference feature.

---

## 05 — ML Pipeline

```text
Raw data
   │
   ├── Remove identifier columns
   │
   ├── Train / test split (80 / 20, stratified)
   │
   ├── IQR clipping on selected numeric variables
   │
   ├── Age / employment sanity filtering
   │
   ├── Binary + ordinal encoding
   │
   ├── One-hot encoding of nominal categories
   │
   ├── RobustScaler fitted on training data only
   │
   ├── SMOTETomek on training data only
   │
   ▼
Tuned XGBoost
   │
   ▼
Held-out test evaluation
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1
   └── Confusion matrix
```

### Preprocessing

The reusable model layer applies the same feature contract during training and inference. This is important because the application should not use a separate hand-written preprocessing path from the one used to train the model.

The pipeline includes:

1. removal of `loan_id` when present;
2. data-dependent IQR clipping on selected numeric variables;
3. sanity filtering for unrealistic age and employment-experience values;
4. binary and ordinal encoding;
5. one-hot encoding for nominal categories;
6. `RobustScaler` transformation;
7. `SMOTETomek` applied only to the training portion;
8. model fitting and held-out evaluation.

### Leakage-aware update

The current implementation fits data-dependent clipping bounds from the **training split only** and uses a **stratified 80/20 split**. This keeps test-set information out of the preprocessing decisions.

---

## 06 — Model Benchmark

The benchmark compared nine classifiers on the held-out test set:

| Model | Accuracy | Class-1 Precision | Class-1 Recall | Class-1 F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.42% | 0.6349 | 0.9155 | 0.7498 |
| KNN | 85.96% | 0.6292 | 0.8975 | 0.7397 |
| Decision Tree | 89.12% | 0.7148 | 0.8495 | 0.7763 |
| Random Forest | 91.64% | 0.7868 | 0.8560 | 0.8199 |
| Extra Trees | 91.28% | 0.7716 | 0.8630 | 0.8147 |
| Gradient Boosting | 90.84% | 0.7618 | 0.8555 | 0.8059 |
| HistGradientBoosting | 93.23% | 0.8866 | 0.7975 | 0.8397 |
| SVM | 88.43% | 0.6809 | 0.9025 | 0.7762 |
| **XGBoost** | **93.47%** | **0.8673** | 0.8335 | **0.8501** |

### Selection logic

XGBoost was selected because it led this benchmark on:

- test accuracy;
- class-1 F1 score.

However, it does **not** dominate every metric. HistGradientBoosting achieved higher class-1 precision, while Logistic Regression and SVM achieved higher class-1 recall. That trade-off is intentionally documented rather than hidden.

> **Metric note:** the 93.47% figure is the documented **current benchmark result** from the nine-model comparison.

---

## 07 — Class Imbalance

The original training distribution was substantially imbalanced:

```text
Before SMOTETomek

Class 0  ████████████████████████████  27,991
Class 1  ████████                       8,001
```

After `SMOTETomek`:

```text
After SMOTETomek

Class 0  ████████████████████████████  27,887
Class 1  ████████████████████████████  27,887
```

The purpose of resampling is to give the learner a more balanced training distribution. Evaluation remains on the held-out test data rather than on the resampled training data.

This is important because a high accuracy value alone can be misleading when one class is less common than the other.

---

## 08 — Model Behavior

### Current XGBoost confusion matrix

```text
                    Predicted
                 0           1
Actual  0      6743        255
        1       333       1667
```

For class 1, the current benchmark reported:

- **Precision:** 0.8673
- **Recall:** 0.8335
- **F1:** 0.8501

This illustrates the precision/recall trade-off. A model can be strong at avoiding false positive class-1 predictions while still missing some true class-1 cases.

The appropriate operating point depends on the real cost of false positives and false negatives. This portfolio project does not define a real lender's business threshold.

---

## 09 — Generalization and Overfitting

The benchmark was not interpreted from training performance alone.

- Decision Tree: **100% training accuracy** vs **89.12% test accuracy** — a clear overfitting signal.
- XGBoost: **97.68% training accuracy** vs **93.47% test accuracy** — a train/test gap exists, but the held-out result remained the strongest benchmark result.

The project therefore avoids describing the model as “perfect” or “production ready.”

The newer XGBoost configuration adds regularization, subsampling and a lower learning rate to target better generalization rather than simply maximizing training performance.

---

## 10 — Feature Representation

The project uses different representations according to the nature of the feature:

- gender as binary encoding;
- previous default history as binary encoding;
- education as an ordinal representation;
- home ownership as one-hot encoding;
- loan intent as one-hot encoding;
- numeric variables passed through the numeric preprocessing pipeline.

### Technical note: ordinal encoding

Ordinal encoding of education introduces an ordering assumption. A future experiment could compare this representation with one-hot encoding to measure whether the assumption affects generalization.

### Technical note: correlation

A near-zero Pearson correlation does **not** automatically mean a feature is useless. Pearson correlation measures a particular form of linear association and should not replace model-based feature analysis.

---

## 11 — Tuned XGBoost Configuration

The current model core uses a more deliberate XGBoost configuration than the original benchmark run:

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

### Why these changes?

- **Lower learning rate + more estimators:** allows more gradual boosting updates.
- **Subsampling:** reduces reliance on the same observations across all trees.
- **Column subsampling:** introduces additional regularization.
- **`min_child_weight` and `gamma`:** make tree growth more conservative.
- **L1/L2 regularization:** penalizes unnecessarily complex solutions.
- **`hist`:** provides a faster histogram-based tree-building method.

The objective is improved generalization and a more controlled training process, not a claim of a guaranteed accuracy increase.

---

## 12 — Application Architecture

```text
┌──────────────────────────────────────────────────────┐
│                 Streamlit Application                │
│                                                      │
│  Prediction Studio │ Batch Lab │ Insights │ Deck    │
└──────────────────────────────┬───────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────┐
│                    src/model.py                      │
│                                                      │
│ train_model()   → LoanModelBundle                   │
│ predict_one()   → single applicant inference         │
│ predict_batch() → CSV / dataframe inference          │
└──────────────────────────────┬───────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────┐
│ Data + preprocessing + scaler + SMOTETomek + XGBoost│
└──────────────────────────────────────────────────────┘
```

### Engineering design principle

```text
Notebook       = research / experimentation
src/           = reusable ML logic
app.py         = product + presentation layer
tests/         = quality contract
docs/          = engineering + model documentation
```

The core exposes a typed `LoanModelBundle` containing the estimator, scaler, feature schema and clipping bounds. This keeps training and inference tied to the same preprocessing contract.

---

## 13 — Interactive Features

### Prediction Studio

The main prediction interface provides:

- the complete applicant input contract;
- numeric controls bounded by dataset-derived ranges;
- categorical inputs matching the training representation;
- real model inference;
- predicted class;
- probability output;
- decision-oriented signals;
- benchmark context;
- session prediction history.

### Batch Lab

```text
CSV template
     ↓
Upload
     ↓
Schema validation
     ↓
Exact preprocessing contract
     ↓
Batch inference
     ↓
Predictions + probabilities
     ↓
Optional labelled-data evaluation
     ↓
Download scored CSV
```

If `loan_status` is present in the upload, the application can also compare predictions against the supplied labels and report uploaded-label accuracy.

### Model Insights

The application surfaces:

- model benchmark metrics;
- comparison table;
- confusion matrix;
- XGBoost feature importance;
- precision/recall trade-offs;
- generalization discussion.

### Project Presentation

The application contains an interactive technical deck covering the actual project workflow, including:

1. project opening and team;
2. problem framing;
3. dataset;
4. data quality;
5. preprocessing;
6. class imbalance;
7. model benchmark;
8. XGBoost selection;
9. generalization and overfitting;
10. product architecture;
11. Prediction Studio;
12. Batch Lab;
13. Model Insights;
14. limitations;
15. production roadmap;
16. team and closing.

---

## 14 — Reproducible Google Colab

The repository includes `Loan_Intelligence_Colab.ipynb` as an additional way to reproduce the ML workflow without local setup.

[Open Loan Intelligence in Google Colab](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)

The notebook:

1. clones the repository;
2. installs the project requirements;
3. loads `loan_data.csv`;
4. imports the reusable functions from `src/model.py`;
5. trains the current model configuration;
6. reports Accuracy, Precision, Recall and F1;
7. displays the confusion matrix;
8. runs a sample applicant through `predict_one()`.

The notebook is a reproducibility and demonstration layer; the main source of truth for model behavior remains the reusable code under `src/`.

---

## 15 — Repository Structure

```text
loan-prediction-nti/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # automated tests and model checks
│       └── pages.yml              # GitHub Pages deployment workflow
│
├── .streamlit/
│   └── config.toml                # Streamlit configuration
│
├── assets/                        # static application / site assets
│
├── docs/
│   ├── architecture.md            # system architecture
│   ├── model-card.md              # model documentation
│   └── quickstart.md              # local / deployment quickstart
│
├── src/
│   ├── __init__.py
│   └── model.py                   # reusable training + inference core
│
├── tests/
│   └── test_model.py              # ML contract tests
│
├── app.py                         # Streamlit application
├── index.html                     # project landing page
├── loan.ipynb                     # original research notebook
├── Loan_Intelligence_Colab.ipynb  # reproducible Colab notebook
├── loan_data.csv                  # project dataset
├── requirements.txt
├── CONTRIBUTING.md
├── .gitignore
└── README.md
```

---

## 16 — Run Locally

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

If PowerShell activation is blocked:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Run the test suite:

```bash
python -m pytest -q
```

Compile the main Python modules:

```bash
python -m compileall app.py src tests
```

The Streamlit application normally opens at:

```text
http://localhost:8501
```

For additional setup details, see `docs/quickstart.md`.

---

## 17 — CI and Engineering Quality

The repository uses GitHub Actions to automate basic quality checks.

The CI workflow:

- installs the project dependencies;
- compiles `app.py`, `src/` and `tests/`;
- runs the automated tests;
- trains the current model configuration;
- reports Accuracy, Precision, Recall, F1, test-row count and feature count;
- prints the confusion matrix for the training run.

This keeps the documented workflow closer to the code that is actually executed rather than relying only on manually copied notebook results.

The project also includes:

- typed model-bundle abstraction;
- reusable single-row inference;
- reusable batch inference;
- schema validation;
- dedicated architecture documentation;
- a model card;
- deployment quickstart;
- explicit responsible-use guidance.

---

## 18 — Deployment

### Streamlit Community Cloud

The repository is structured for deployment through Streamlit Community Cloud:

```text
Repository: engyusufayman06/loan-prediction-nti
Branch:     main
Main file:  app.py
```

The application reads `loan_data.csv` from the repository root and uses the reusable model layer for inference.

### GitHub Pages

The repository also contains a lightweight project landing page:

```text
https://engyusufayman06.github.io/loan-prediction-nti/
```

Deployment is handled through the repository's GitHub Actions Pages workflow.

---

## 19 — Limitations and Responsible Use

This system is an **educational / portfolio ML demonstration**.

It has not been validated as a real lender's underwriting model and should not be used as the sole basis for a real financial decision.

Known limitations include:

- the dataset may not represent a real lender's current population;
- no complete fairness or subgroup-performance audit has been completed;
- probability calibration has not been established for real-world approval odds;
- dataset shift and production drift are not monitored;
- the benchmark does not establish causal relationships;
- class balancing changes the training distribution;
- the current preprocessing workflow should be hardened further for strict production validation;
- production monitoring, governance and model registry are not included.

**Never use this model as the sole basis for a real financial decision.**

---

## 20 — Production Roadmap

```text
CURRENT PORTFOLIO SYSTEM
        │
        ├── Interactive Streamlit application
        ├── Reusable ML core
        ├── Batch inference
        ├── Model diagnostics
        └── Automated tests / CI
        │
        ▼
NEXT ITERATION
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

The roadmap separates what is already implemented from what would be required to move from an educational portfolio project toward a production-grade ML service.

---

## 21 — Team

### NTI Machine Learning Track

| # | Team member |
|---:|---|
| 01 | **Yusuf Ayman Tolba** |
| 02 | **Mohamed Reda Hussein** |
| 03 | **Abdelrahman Mohamed Ahmed** |
| 04 | **Abdelmoniem Ibrahim Abdelmoniem** |
| 05 | **Asmaa Rabee Mohammed** |

---

## 22 — Project Links

- **GitHub Repository:** https://github.com/engyusufayman06/loan-prediction-nti
- **Google Colab:** https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb
- **GitHub Pages:** https://engyusufayman06.github.io/loan-prediction-nti/
- **NTI:** https://www.nti.sci.eg/

---

## License

Educational NTI project. Add an appropriate open-source license before redistributing the dataset or code outside the course context.

<div align="center">

**Built for the NTI Machine Learning Track**  
**Loan Intelligence · End-to-End Machine Learning Project**

</div>
