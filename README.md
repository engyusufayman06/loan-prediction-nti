<div align="center">

# ◈ LOAN INTELLIGENCE
### NTI Machine Learning Track · End-to-End Loan Status Classification

**From raw applicant data → reproducible ML pipeline → engineered features → benchmarked models → interactive decision studio → deployable demo.**

<br>

<img src="https://img.shields.io/badge/Track-Machine%20Learning-06B6D4?style=for-the-badge">
<img src="https://img.shields.io/badge/Model-HistGradientBoosting-7C3AED?style=for-the-badge">
<img src="https://img.shields.io/badge/Benchmark%20Leader-HistGradientBoosting-22C55E?style=for-the-badge">
<img src="https://img.shields.io/badge/HistGradientBoosting%20Test%20Accuracy-93.29%25-22C55E?style=for-the-badge">
<img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">

<br><br>

<a href="https://github.com/engyusufayman06/loan-prediction-nti">Repository</a> ·
<a href="https://www.nti.sci.eg/">NTI</a>

</div>

---

## ⚡ The project in 30 seconds

**Loan Intelligence** is an educational machine-learning system for **binary loan-status classification**. Instead of stopping at a notebook, the project packages the experiment into a small product: a reusable preprocessing/model layer, an interactive Streamlit application, batch CSV inference, model diagnostics, feature engineering, benchmarking, and an in-app technical presentation.

> **Final notebook benchmark:** HistGradientBoosting reached **93.2874% held-out accuracy**, **0.860537 precision**, **0.8330 recall**, **0.846545 F1**, and **0.976527 ROC-AUC**. It is the repository's benchmark leader and deployable model.

This repository is intentionally transparent: metrics, preprocessing decisions, engineered features, class balancing, model trade-offs, limitations, feature attribution, and the separate XGBoost optimization experiment are documented rather than hidden behind a single accuracy number.

---

## 🎬 Experience the system

The Streamlit app is organized as a product workspace:

| Workspace | What it does |
|---|---|
| **Prediction Studio** | Enter a complete applicant profile and run HistGradientBoosting inference. |
| **Batch Lab** | Upload a CSV, validate its schema, score rows, inspect results, and download predictions. |
| **Model Insights** | Explore the complete nine-model benchmark, ROC-AUC, confusion matrix, HistGradientBoosting diagnostics, feature importance, dataset statistics, and research experiments. |
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
                 │ → HistGradientBoosting         │
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
- reusable inference;
- model interpretation.

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
   └── model-specific scaling
          ├── Logistic Regression / KNN / SVM → StandardScaler
          └── Tree models → unscaled engineered features
   │
   ▼
HistGradientBoosting deployable inference model
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

The important engineering step is consistency. The same feature contract and transformations are reused during inference so that a prediction in the UI follows the project's trained representation rather than a separate, hand-written preprocessing path.

---

# 04 · Model benchmark

The final notebook evaluates **nine classifiers** on the same stratified held-out test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **HistGradientBoosting** | **93.2874%** | **0.860537** | 0.8330 | **0.846545** | **0.976527** |
| **XGBoost** | 92.9651% | 0.855434 | 0.8225 | 0.838644 | 0.975957 |
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

The repository uses HistGradientBoosting as the deployable model because it leads the final benchmark by F1, accuracy, and ROC-AUC. XGBoost remains documented as the close second-place benchmark model.

That distinction is deliberate: the README keeps the complete benchmark instead of deleting the other experiments just because one model was selected for deployment.

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

### HistGradientBoosting confusion matrix

```text
                    Predicted
                 0           1
Actual  0      6728        270
        1       334       1666
```

From this matrix:

- True Negatives: **6,728**
- False Positives: **270**
- False Negatives: **334**
- True Positives: **1,666**
- Error rate: **6.7126%**
- Correct prediction rate: **93.2874%**
- False Positive Rate: **3.8582%**
- False Negative Rate: **16.7000%**

### Full classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Rejected (0) | **0.952705** | **0.961418** | **0.957041** | **6,998** |
| Approved (1) | **0.860537** | **0.833000** | **0.846545** | **2,000** |
| Macro avg | 0.906621 | 0.897209 | 0.901793 | 8,998 |
| Weighted avg | 0.932218 | 0.932874 | 0.932481 | 8,998 |

The positive-class recall means the model identifies **1,666 of 2,000** approved examples in the held-out test set. The operating point should be interpreted in terms of the relative cost of false positives and false negatives, not accuracy alone.

---

# 07 · Generalization & overfitting

The final notebook includes a **learning-curve assessment** using three-fold stratified cross-validation and F1 as the scoring metric. It also compares all nine candidate models on the same held-out split.

The project therefore treats generalization as a separate question from raw training performance. The repository does **not** claim that a 93% test accuracy automatically means production readiness.

Known facts from the benchmark:

- HistGradientBoosting: **93.2874%** held-out accuracy.
- XGBoost: **92.9651%** held-out accuracy.
- Random Forest: **91.1202%** held-out accuracy.
- Logistic Regression: **86.1747%** held-out accuracy.

The spread demonstrates why the project includes model comparison, learning curves, ROC/PR analysis, and explicit limitations rather than presenting one algorithm in isolation.

---

# 08 · Feature representation

The final notebook transforms the raw schema into an **18-feature** model representation.

It uses:

- LabelEncoder for object/categorical columns;
- five engineered ratio/indicator features;
- a stratified 80/20 split;
- SMOTE for the training split;
- StandardScaler for the scale-sensitive benchmark models;
- the unscaled engineered representation for tree models including HistGradientBoosting.

### Top HistGradientBoosting features

HistGradientBoosting does not expose native `feature_importances_`. The notebook's global feature-importance cross-examination therefore uses **permutation importance** for this model. The values below are normalized relative importance percentages from five repeats on the first 1,000 rows of the held-out test set.

| Rank | Feature | Relative importance |
|---:|---|---:|
| 1 | `previous_loan_defaults_on_file` | **31.3320%** |
| 2 | `person_income` | **11.5035%** |
| 3 | `income_to_loan_ratio` | **9.4854%** |
| 4 | `person_emp_exp` | **8.5267%** |
| 5 | `loan_int_rate` | **8.4258%** |
| 6 | `emp_length_to_age_ratio` | **6.9122%** |
| 7 | `person_home_ownership` | **6.8618%** |
| 8 | `loan_intent` | **3.9354%** |
| 9 | `cb_person_cred_hist_length` | **2.6236%** |
| 10 | `credit_score` | **2.3209%** |

The ranking is a model-attribution result, not a causal statement. A larger percentage means that shuffling that feature affected the measured model performance more strongly in this experiment; it does not mean the feature alone determines whether a person should receive a loan.

The full notebook also performs feature-importance cross-examination across the nine models, using native importance where available and permutation importance where native importance is not exposed.

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
│ HistGradientBoosting inference                       │
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

Enter the complete 13-field applicant input contract. The application automatically creates the five engineered features used by the final notebook and sends the row through the reusable preprocessing/model layer with HistGradientBoosting as the deployable classifier.

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

The batch interface uses the same raw schema as the single-applicant workflow, so the demo is not using a separate simplified model.

### Model Insights

The application surfaces:

- all nine benchmark models;
- Accuracy, Precision, Recall, F1 and ROC-AUC;
- dataset and split statistics;
- SMOTE balancing statistics;
- HistGradientBoosting confusion matrix;
- FPR / FNR / error rate / correct rate;
- full HistGradientBoosting classification report;
- top HistGradientBoosting permutation features;
- model configuration;
- the notebook's separate XGBoost hyperparameter-search experiment.

### Demo CSV

`demo_test.csv` contains six valid applicant records covering different education, income, home ownership, loan intent, credit-score, and previous-default scenarios. The current HistGradientBoosting demo produces both approval and rejection outcomes across the supplied records.

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

```text
Repository: engyusufayman06/loan-prediction-nti
Branch:     main
Main file:  app.py
```

The application reads `loan_data.csv` from the repository root and caches the trained HistGradientBoosting model during normal Streamlit interaction. No separate backend service is required for this educational demo.

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
- a dedicated benchmark script;
- architecture documentation;
- a model card;
- an explicit nine-model benchmark;
- model diagnostics and feature attribution;
- a demo CSV;
- explicit limitations and responsible-use guidance.

The key engineering rule is that the application consumes the same feature contract as the research pipeline. This reduces the common notebook-to-demo mismatch where the model is trained one way and deployed another way.

---

# 15 · Limitations & responsible use

This system is an **educational / portfolio ML demonstration**. It has not been validated as a real lender's underwriting model and should not be used as the sole basis for a real financial decision.

Known limitations include:

- the dataset may not represent a real lender's current population;
- categorical encoding is learned from the supplied dataset;
- SMOTE changes the training distribution;
- subgroup fairness has not been fully audited;
- probability calibration has not been established for real-world approval odds;
- dataset shift and production drift are not monitored;
- benchmark metrics do not establish causal relationships;
- feature importance is model attribution, not causal inference;
- the separate XGBoost optimization experiment is a research result and is not the deployed model.

The demo's approval/rejection probabilities should therefore be read as **model scores**, not guaranteed real-world probabilities.

---

# 16 · Production roadmap

```text
CURRENT
  │
  ├── Interactive Streamlit application
  ├── Reusable HistGradientBoosting ML core
  ├── Batch inference
  ├── Engineered features
  ├── Automated tests / CI
  └── Nine-model benchmark

NEXT
  │
  ├── Versioned model artifact
  ├── FastAPI inference service
  ├── SHAP local explanations
  ├── Probability calibration
  ├── Fairness / subgroup evaluation
  ├── Dockerized deployment
  └── Data / concept drift monitoring

PRODUCTION MATURITY
  │
  ├── MLflow experiment tracking
  ├── Model registry
  ├── Data / concept drift monitoring
  ├── Observability + alerting
  └── Reproducible CI/CD
```

The roadmap is deliberately separated from the current capabilities so future engineering work is not confused with functionality that already exists.

---

# 17 · Team

### NTI Machine Learning Track

| # | Team member |
|---|---|
| 01 | **Yusuf Ayman Tolba** |
| 02 | **Mohamed Reda Hussein** |
| 03 | **Abdelrahman Mohamed Ahmed** |
| 04 | **Abdelmoniem Ibrahim Abdelmoniem** |
| 05 | **Asmaa Rabee Mohammed** |

The project was developed as a team-based NTI machine-learning project, combining research, modeling, application development, testing, documentation, and presentation work into one deliverable.

---

# 18 · NTI

This project was developed in the context of the **National Telecommunication Institute (NTI) Machine Learning Track**.

The repository demonstrates the complete lifecycle of an educational ML project: understanding the dataset, preparing the representation, engineering features, comparing algorithms, validating the chosen model, interpreting model behavior, and shipping the result as an interactive application.

**Official NTI:** https://www.nti.sci.eg/

---

## Final note

Loan Intelligence is intentionally more than a classifier. The repository is a compact example of how a machine-learning experiment can move from **data → preprocessing → feature engineering → imbalance handling → benchmark → model selection → interpretation → reusable inference → interactive product** without hiding the experimental details.

The headline number is **93.2874% held-out accuracy for HistGradientBoosting**, but the more important result is the reproducible workflow around it.
