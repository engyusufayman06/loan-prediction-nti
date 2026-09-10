<div align="center">

# ◈ LOAN INTELLIGENCE
### NTI Machine Learning Track · End-to-End Loan Status Classification

**From raw applicant data → reproducible ML pipeline → benchmarked models → interactive decision studio → deployable demo.**

<br>

<img src="https://img.shields.io/badge/Track-Machine%20Learning-06B6D4?style=for-the-badge">
<img src="https://img.shields.io/badge/Model-XGBoost-7C3AED?style=for-the-badge">
<img src="https://img.shields.io/badge/Test%20Accuracy-93.28%25-22C55E?style=for-the-badge">
<img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">

<br><br>

<a href="https://github.com/engyusufayman06/loan-prediction-nti">Repository</a> ·
<a href="https://www.nti.sci.eg/">NTI</a>

</div>

---

## ⚡ The project in 30 seconds

**Loan Intelligence** is an educational machine-learning system for **binary loan-status classification**. Instead of stopping at a notebook, the project packages the experiment into a small product: a reusable preprocessing/model layer, an interactive Streamlit application, batch CSV inference, model diagnostics, and an in-app technical presentation.

> **Headline result:** XGBoost reached **93.28% held-out test accuracy**, with **0.8700 class-1 precision** and **0.8443 class-1 F1** in the project's nine-model benchmark.

This repository is intentionally transparent: metrics, preprocessing decisions, class balancing, trade-offs, limitations, and the path toward production are documented rather than hidden behind a single accuracy number.

---

## 🎬 Experience the system

The Streamlit app is organized as a product workspace:

| Workspace | What it does |
|---|---|
| **Prediction Studio** | Enter a complete applicant profile and run real XGBoost inference. |
| **Batch Lab** | Upload a CSV, validate its schema, score rows, inspect results, and download predictions. |
| **Project Presentation** | Navigate a fullscreen, interactive technical deck covering the complete project story. |
| **Model Insights** | Explore the nine-model benchmark, XGBoost feature importance, confusion matrix, and metric trade-offs. |
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
                 │      Reusable ML Core          │
                 │  cleaning → encoding → scale  │
                 │       → SMOTETomek → XGB      │
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
                    │ Interactive Presentation│
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
| Rows after preprocessing | **44,990** |
| Encoded model features | **19** |
| Target | `loan_status` |
| Original class 0 | **35,000** |
| Original class 1 | **10,000** |
| Post-SMOTETomek class 0 | **27,899** |
| Post-SMOTETomek class 1 | **27,899** |

The notebook identifies the dataset source as Kaggle. The exact dataset URL is not recorded in the project files, so this README deliberately does **not** fabricate one.

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

```text
Raw data
   │
   ├── remove loan_id
   │
   ├── IQR clipping on selected numeric variables
   │
   ├── age / employment sanity filters
   │
   ├── binary + ordinal encoding
   │
   ├── one-hot encoding of nominal categories
   │
   ├── 80/20 train-test split (random_state=42)
   │
   ├── RobustScaler fitted on training data
   │
   ├── SMOTETomek on scaled training data
   │
   └── model benchmarking
   │
   ▼
XGBoost selected from the benchmark
```

### Why the pipeline matters

The important engineering step is not simply training XGBoost. The same feature contract and transformations are reused during inference so that a prediction in the UI follows the project's trained representation rather than a separate, hand-written preprocessing path.

---

# 04 · Model benchmark

| Model | Accuracy | Class-1 Precision | Class-1 Recall | Class-1 F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.40% | 0.6344 | **0.9155** | 0.7495 |
| KNN | 86.52% | 0.6463 | 0.8690 | 0.7413 |
| Decision Tree | 88.58% | 0.6935 | 0.8710 | 0.7722 |
| Random Forest | 92.01% | 0.8111 | 0.8350 | 0.8229 |
| Extra Trees | 91.68% | 0.8000 | 0.8340 | 0.8166 |
| Gradient Boosting | 90.08% | 0.7311 | 0.8755 | 0.7968 |
| HistGradientBoosting | 91.71% | 0.7946 | 0.8455 | 0.8193 |
| SVM | 88.19% | 0.6727 | **0.9125** | 0.7745 |
| **XGBoost** | **93.28%** | **0.8700** | 0.8200 | **0.8443** |

### Selection logic

XGBoost is selected for this project because it leads the benchmark on **test accuracy, class-1 precision, and class-1 F1**.

However, it does **not** dominate every metric: Logistic Regression and SVM reach higher class-1 recall. That trade-off is intentionally preserved in the documentation and UI rather than hidden.

---

# 05 · What happens to class imbalance?

Before resampling, the training data contained:

```text
Class 0  ████████████████████████████  27,992
Class 1  ████████                       8,000
```

After SMOTETomek:

```text
Class 0  ████████████████████████████  27,899
Class 1  ████████████████████████████  27,899
```

The goal is to make the learner see a more balanced training distribution. Evaluation remains on the held-out test set.

---

# 06 · Model behavior, not just accuracy

### XGBoost confusion matrix

```text
                    Predicted
                 0           1
Actual  0      6753        245
        1       360       1640
```

For class 1, the benchmark reports:

- **Precision:** 0.8700
- **Recall:** 0.8200
- **F1:** 0.8443

This means the selected model is strong at avoiding false positive class-1 predictions relative to the other benchmarked models, while still missing some true class-1 cases. The right operating point depends on the business cost of false positives versus false negatives.

---

# 07 · Generalization & overfitting

The benchmark was not interpreted from training performance alone.

- Decision Tree: **88.58% test accuracy** → lower than XGBoost in the held-out benchmark.
- XGBoost: **93.28% test accuracy** → the strongest held-out benchmark result.

The README therefore avoids calling the model “perfect” or “production ready.”

---

# 08 · Feature representation

The project encodes:

- gender as binary;
- previous default history as binary;
- education as an ordinal representation;
- home ownership and loan intent using one-hot encoding.

> **Technical note:** ordinal encoding of education introduces an ordering assumption. A future experiment can compare it against one-hot encoding to test whether that assumption affects generalization.

Also, a near-zero Pearson correlation does **not** automatically mean a feature is useless: correlation captures a particular form of linear association and does not replace model-based analysis.

---

# 09 · Application architecture

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
│ train_model()  →  LoanModelBundle                   │
│ predict_one()  →  single applicant inference        │
│ predict_batch()→  CSV / dataframe inference         │
└──────────────────────────────┬───────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────┐
│ Data + preprocessing + scaler + SMOTETomek + XGBoost│
└──────────────────────────────────────────────────────┘
```

### Design principle

**Notebook = research.**  
**`src/` = reusable ML logic.**  
**`app.py` = product/presentation layer.**  
**`tests/` = quality contract.**  
**`docs/` = engineering and model documentation.**

---

# 10 · Repository map

```text
loan-prediction-nti/
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # automated checks
│
├── .streamlit/
│   └── config.toml                # application theme/config
│
├── docs/
│   ├── architecture.md            # system architecture
│   ├── model-card.md              # model documentation
│   └── quickstart.md              # deployment/local quickstart
│
├── src/
│   ├── __init__.py
│   └── model.py                   # training + inference core
│
├── tests/
│   └── test_model.py              # ML contract tests
│
├── app.py                         # interactive product UI
├── loan.ipynb                     # original research notebook
├── loan_data.csv                  # project dataset
├── requirements.txt
├── CONTRIBUTING.md
├── .gitignore
└── README.md
```

---

# 11 · Interactive features

### Prediction Studio

Enter the full applicant input contract. Numeric fields expose dataset-derived bounds, then the application sends the row through the same reusable preprocessing/model layer.

### Batch Lab

```text
CSV template
     ↓
Upload
     ↓
Schema validation
     ↓
Batch inference
     ↓
Predictions + probabilities
     ↓
Optional labelled-data evaluation
     ↓
Download scored CSV
```

### Project Presentation

The application contains an interactive technical deck with fullscreen presentation mode, slide navigation, keyboard controls, and a narrative that follows the actual project workflow.

### Model Insights

The application surfaces:

- benchmark metrics;
- confusion matrix;
- XGBoost feature importance;
- class-1 precision/recall trade-off;
- generalization discussion.

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

More details are available in [`docs/quickstart.md`](docs/quickstart.md).

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
- unit tests;
- GitHub Actions CI;
- dedicated architecture documentation;
- a model card;
- deployment quickstart;
- explicit limitations and responsible-use guidance.

---

# 15 · Limitations & responsible use

This system is an **educational / portfolio ML demonstration**.

It has not been validated as a real lender's underwriting model and should not be used as the sole basis for a real financial decision.

Known limitations include:

- the dataset may not represent a real lender's current population;
- no fairness audit or subgroup performance analysis has been completed;
- probability calibration has not been established for real-world approval odds;
- dataset shift and production drift are not monitored;
- the benchmark does not establish causal relationships;
- class balancing changes the training distribution;
- the current preprocessing implementation preserves the notebook's IQR-bound workflow and should be hardened further for strict production validation.

---

# 16 · Production roadmap

```text
CURRENT
  │
  ├── Interactive Streamlit application
  ├── Reusable ML core
  ├── Batch inference
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

---

<div align="center">

### Built to demonstrate the full ML journey — not just the final score.

**Research → Engineering → Evaluation → Product → Deployment**

</div>
