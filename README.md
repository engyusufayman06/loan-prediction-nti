<div align="center">

# ◈ LOAN INTELLIGENCE
### NTI Machine Learning Track · End-to-End Loan Status Classification

**From raw applicant data → reproducible ML pipeline → tuned XGBoost → interactive decision studio.**

[![Model](https://img.shields.io/badge/Model-Tuned%20XGBoost-7C3AED?style=for-the-badge)](src/model.py)
[![App](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](app.py)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions)](.github/workflows)
[![Notebook](https://img.shields.io/badge/Notebook-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)

### [▶ Open the Colab notebook](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb) · [Repository](https://github.com/engyusufayman06/loan-prediction-nti)

</div>

---

## ⚡ What is this?

**Loan Intelligence** is an educational machine-learning project that predicts `loan_status` from applicant and loan characteristics.

The project is built as a small end-to-end ML product instead of a notebook-only assignment:

- reusable training and inference code in `src/`;
- tuned XGBoost classification;
- class-imbalance handling with SMOTETomek;
- a polished Streamlit Prediction Studio;
- CSV Batch Lab for multiple applicants;
- model diagnostics and technical presentation;
- reproducible Google Colab notebook;
- automated tests and GitHub Actions.

> **Important:** this is an educational / portfolio project, not a real credit-underwriting system. Model probabilities should not be interpreted as real-world approval odds.

---

## 🚀 Run it instantly in Colab

No local setup is required.

**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engyusufayman06/loan-prediction-nti/blob/main/Loan_Intelligence_Colab.ipynb)**

The notebook automatically:

1. clones this repository;
2. installs the pinned dependencies;
3. loads `loan_data.csv`;
4. trains the same model used by the application;
5. prints Accuracy / Precision / Recall / F1;
6. displays the confusion matrix;
7. runs a sample applicant through the real inference function.

This makes the project easy to demonstrate during an NTI presentation or interview.

---

## 🎬 Project experience

| Workspace | Purpose |
|---|---|
| **Prediction Studio** | Enter one applicant and receive an approval/rejection prediction with probabilities. |
| **Batch Lab** | Upload a CSV, validate the schema, score many applicants, evaluate labelled data, and download results. |
| **Model Insights** | Inspect benchmark metrics, confusion matrix, feature importance, and trade-offs. |
| **Project Presentation** | Follow the full project story inside the app with an interactive technical deck. |
| **Colab** | Reproduce the ML experiment directly in the browser. |

### Product flow

```text
                       ┌──────────────────────┐
                       │    loan_data.csv     │
                       └──────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Cleaning + Validation  │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ Encoding + Scaling     │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │     SMOTETomek         │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │   Tuned XGBoost        │
                     └───────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       Prediction Studio      Batch Lab       Model Insights
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                       Interactive Demo
```

---

## 🧠 ML pipeline

The current training pipeline uses a clean train/test separation and keeps preprocessing decisions inside the reusable model layer.

```text
Raw dataset
    │
    ├── remove loan_id
    ├── train/test split (80/20, stratified)
    ├── IQR clipping fitted from training data
    ├── age / employment sanity filtering
    ├── binary + ordinal encoding
    ├── one-hot encoding of nominal categories
    ├── RobustScaler fitted on training data
    ├── SMOTETomek on training data only
    │
    ▼
Tuned XGBoost
    │
    ▼
Held-out test evaluation
```

### Why the update matters

The original baseline used a default XGBoost configuration and a non-stratified split. The current implementation improves the experiment by:

- using a **stratified 80/20 split**;
- fitting clipping bounds from **training data only**;
- preserving the same preprocessing contract at inference time;
- using a lower learning rate with more boosting rounds;
- adding subsampling and regularization to control overfitting;
- using XGBoost's `hist` tree method for faster training.

The app calculates the current metrics at runtime instead of hard-coding a new accuracy number before the tuned model is evaluated.

### Tuned XGBoost configuration

```text
n_estimators      = 600
max_depth         = 5
learning_rate     = 0.05
subsample         = 0.90
colsample_bytree  = 0.90
min_child_weight  = 2
gamma             = 0.05
reg_alpha         = 0.05
reg_lambda        = 2.0
```

The goal is **better generalization**, not simply maximizing a training score.

---

## 📊 Dataset

The repository contains `loan_data.csv` with **45,000 rows** and the following original columns:

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

### Target

`loan_status` is the binary classification target.

### Model inputs

The model uses 13 applicant/loan attributes. `loan_id`, when present in a batch CSV, is treated only as an identifier and is never used as a feature.

---

## 🔬 Evaluation

The project reports more than accuracy because loan classification has an important precision/recall trade-off.

The application displays:

- **Accuracy** — overall fraction of correct predictions;
- **Precision** — how reliable positive predictions are;
- **Recall** — how many true positive cases are found;
- **F1** — balance between precision and recall;
- **Confusion Matrix** — TP / TN / FP / FN behavior.

The previous baseline reached **92.87% held-out accuracy**. The current tuned pipeline is evaluated independently at runtime, so the README does not invent a new headline score before the new configuration has actually been executed.

---

## 🎨 Streamlit demo

The application is designed as a product-style ML workspace rather than a default Streamlit form.

### Prediction Studio

- dark technical visual system;
- structured applicant input;
- dataset-aware numeric controls;
- immediate prediction and probability output;
- clear approved/rejected result state.

### Batch Lab

```text
CSV template
     ↓
Upload
     ↓
Schema validation
     ↓
Exact model inference pipeline
     ↓
Predictions + probabilities
     ↓
Optional labelled-data evaluation
     ↓
Download scored CSV
```

### Model Insights

The app exposes the model's benchmark and diagnostic information so the demo explains **why the model was selected**, not only what prediction it produced.

---

## 🛠️ Run locally

### Windows

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

Run tests:

```bash
python -m pytest -q
```

---

## 📁 Repository structure

```text
loan-prediction-nti/
│
├── .github/workflows/       # CI + GitHub Pages
├── .streamlit/              # Streamlit configuration
├── assets/                  # Static website assets
├── docs/                    # Architecture, model card, quickstart
├── src/
│   ├── __init__.py
│   └── model.py             # reusable training + inference core
├── tests/                   # model contract tests
│
├── app.py                   # Streamlit application
├── index.html                # GitHub Pages project landing page
├── loan.ipynb                # original research notebook
├── Loan_Intelligence_Colab.ipynb  # one-click reproducible Colab
├── loan_data.csv             # dataset
├── requirements.txt
└── README.md
```

### Engineering principle

**Notebook = research**  
**`src/` = reusable ML logic**  
**`app.py` = product / presentation layer**  
**`tests/` = quality contract**  
**`docs/` = engineering documentation**

---

## 🌐 Deployment

### Streamlit Community Cloud

Use:

```text
Repository: engyusufayman06/loan-prediction-nti
Branch:     main
Main file:  app.py
```

### GitHub Pages

The repository also contains a lightweight project landing page and a GitHub Actions workflow for Pages deployment.

Project site:

**https://engyusufayman06.github.io/loan-prediction-nti/**

---

## ⚠️ Limitations & responsible use

This is an **educational / portfolio ML demonstration**.

It has not been validated as a real lender's underwriting model and should not be used as the sole basis for a real financial decision.

Known limitations:

- dataset shift is not monitored;
- no production fairness audit has been completed;
- probability calibration has not been established for real approval odds;
- no causal interpretation is intended;
- SMOTETomek changes the training distribution;
- subgroup performance should be evaluated before any real-world use.

---

## 👥 Team

### NTI Machine Learning Track

| # | Team member |
|---:|---|
| 01 | **Yusuf Ayman Tolba** |
| 02 | **Mohamed Reda Hussein** |
| 03 | **Abdelrahman Mohamed Ahmed** |
| 04 | **Abdelmoniem Ibrahim Abdelmoniem** |
| 05 | **Asmaa Rabee Mohammed** |

---

<div align="center">

**Built for NTI Machine Learning Track**  
**Loan Intelligence · End-to-End ML Project**

</div>
