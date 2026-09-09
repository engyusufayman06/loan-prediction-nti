# ◈ Loan Intelligence — NTI Machine Learning Project

<p align="center">
  <strong>End-to-end Machine Learning system for loan-status prediction</strong><br>
  Research notebook • reusable inference • model benchmarking • interactive product UI • batch CSV scoring
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Track" src="https://img.shields.io/badge/Track-Machine%20Learning-0EA5E9">
  <img alt="Model" src="https://img.shields.io/badge/Selected%20model-XGBoost-7C3AED">
  <img alt="Accuracy" src="https://img.shields.io/badge/Test%20accuracy-92.87%25-22C55E">
  <img alt="UI" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
</p>

> **Loan Intelligence** turns the NTI Machine Learning project into a complete demonstrable ML product: experimentation, reusable preprocessing, model benchmarking, single-applicant inference, batch CSV scoring, model diagnostics and an in-app technical presentation.

---

## 01 — Product architecture

```text
loan_data.csv
     │
     ▼
Research / EDA / notebook
     │
     ▼
Reusable ML core — src/model.py
     │
     ├───────────────┬────────────────┐
     ▼               ▼                ▼
Prediction Studio  Batch Lab     Model Insights
     │               │                │
     └───────────────┴────────────────┘
                     ▼
          In-app Project Presentation
```

The repository deliberately separates **research**, **reusable ML logic**, **product UI**, **tests**, and **documentation**.

## 02 — Results

Six classifiers were benchmarked on the same held-out test setup:

| Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.55% | 0.64 | **0.92** | 0.75 |
| KNN | 86.29% | 0.64 | 0.88 | 0.74 |
| Decision Tree | 89.28% | 0.73 | 0.82 | 0.77 |
| Random Forest | 89.48% | 0.71 | 0.88 | 0.79 |
| SVM | 88.02% | 0.67 | **0.92** | 0.77 |
| **XGBoost** | **92.87%** | **0.87** | 0.80 | **0.83** |

XGBoost is the selected benchmark model because it achieved the strongest combination of test accuracy, class-1 precision and class-1 F1 in this comparison. Logistic Regression and SVM achieved higher class-1 recall, so the selection is a **trade-off**, not a universal claim.

## 03 — Dataset & raw inference contract

The repository contains `loan_data.csv` used by the notebook and application.

- **Rows after preprocessing:** 44,990
- **Processed model features:** 20
- **Target:** `loan_status`
- **Original training distribution:** 27,991 class-0 / 8,001 class-1
- **After SMOTETomek:** 27,887 / 27,887

The application accepts these raw applicant fields:

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

`loan_id` is optional for batch uploads. `loan_status` is optional and is used only when evaluating an uploaded labelled CSV.

The notebook identifies the dataset source as Kaggle, but the exact Kaggle URL is not recorded in the project files, so no URL is fabricated here.

## 04 — Machine Learning pipeline

1. Remove `loan_id` as an identifier.
2. Clip selected numeric variables using IQR bounds.
3. Filter unrealistic `person_age` and `person_emp_exp` values.
4. Encode binary/ordinal variables and one-hot encode nominal categories.
5. Split train/test with `random_state=42`.
6. Fit `RobustScaler` on the training split.
7. Apply `SMOTETomek` to the scaled training data only.
8. Train and benchmark six classifiers.
9. Select XGBoost from the benchmark.
10. Reuse the same preprocessing contract for single-row and batch inference.

## 05 — Product UI

### Prediction Studio

A dark FinTech-style workspace with:

- Every raw model input visible
- Dataset-derived numeric min/max ranges
- Applicant profile + credit profile sections
- Real XGBoost inference
- Approval probability
- Decision signals
- Benchmark context
- Session prediction history

### Batch Lab

The app can now work with a real CSV workflow:

```text
Download template
      ↓
Fill applicant rows
      ↓
Upload CSV
      ↓
Schema validation
      ↓
XGBoost batch inference
      ↓
Inspect results
      ↓
Download scored CSV
```

If the uploaded CSV contains `loan_status`, the UI also reports row-level correctness and aggregate accuracy against those supplied labels.

Streamlit provides the file-upload and generated-file download primitives used by this interface. urlStreamlit file uploader documentationhttps://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader urlStreamlit download button documentationhttps://docs.streamlit.io/develop/api-reference/widgets/st.download_button

### Project Presentation

The UI contains a **16-slide technical + presentation deck** covering the complete project story:

1. Project opening / team
2. Problem framing
3. Dataset
4. Data quality
5. Preprocessing
6. Class imbalance
7. Model benchmark
8. XGBoost selection
9. Generalization and overfitting awareness
10. Product architecture
11. Prediction Studio
12. Batch Lab
13. Model Insights
14. Limitations
15. Production roadmap
16. Team / closing

### Model Insights

- Benchmark chart
- Full comparison table
- XGBoost feature importance
- Confusion matrix
- Precision/recall trade-off
- Generalization discussion

### Team & About

The application includes team credits and an official NTI branding reference. The official NTI website describes NTI as an Egyptian center of excellence in education, applied research and technical consultation, established in 1983. citeturn1search0turn1search3

## 06 — Repository structure

```text
loan-prediction-nti/
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── assets/README.md
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

### Engineering principle

```text
Notebook       = research / experimentation
src/           = reusable ML logic
app.py         = product + presentation layer
tests/         = quality checks
.github/       = CI automation
docs/          = architecture + model governance
```

## 07 — Local run

```bash
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, normally `http://localhost:8501`.

## 08 — Engineering quality

The reusable model layer exposes a `LoanModelBundle` containing the trained estimator, scaler, feature schema and clipping bounds. `predict_one()` handles single applicant inference and `predict_batch()` handles CSV-style batch inference with the same preprocessing contract.

The repository also includes unit tests, GitHub Actions CI, Streamlit theme configuration, architecture documentation and a model card.

## 09 — Limitations & responsible use

This is an **educational / portfolio ML demonstration**, not a production credit-underwriting system.

Known limitations:

- The dataset may not represent a real lender's current applicant population.
- Correlation is not a complete measure of feature usefulness.
- Class balancing changes the training distribution.
- The benchmark has not been audited for fairness, calibration or dataset shift.
- No real-time data integration is included.
- Model probabilities should not be interpreted as guaranteed real-world approval odds.

**Never use this model as the sole basis for a real financial decision.**

## 10 — Production roadmap

```text
Current portfolio system
        ↓
Versioned model artifact
        ↓
FastAPI inference service
        ↓
SHAP local explanations
        ↓
Probability calibration
        ↓
Fairness / subgroup evaluation
        ↓
Docker + CI/CD
        ↓
MLflow experiment tracking
        ↓
Monitoring + drift detection
```

## 11 — Team

**NTI Machine Learning Track**

1. **Yusuf Ayman Tolba**
2. **Mohamed Reda Hussein**
3. **Abdelrahman Mohamed Ahmed**
4. **Abdelmoniem Ibrahim Abdelmoniem**

## 12 — Official NTI reference

urlNational Telecommunication Institute (NTI)https://www.nti.sci.eg/

## License

This repository is intended as an educational NTI project. Add the appropriate license before redistributing the dataset or code outside the course context.
