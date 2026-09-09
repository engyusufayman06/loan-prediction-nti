# ◈ Loan Intelligence — NTI Machine Learning Project

<p align="center">
<strong>End-to-end Machine Learning system for loan-status prediction</strong><br>
Research • preprocessing • class balancing • model benchmarking • interactive inference • batch CSV scoring
</p>

<p align="center">
<img src="https://img.shields.io/badge/Track-Machine%20Learning-0EA5E9">
<img src="https://img.shields.io/badge/Selected%20Model-XGBoost-7C3AED">
<img src="https://img.shields.io/badge/Test%20Accuracy-92.87%25-22C55E">
<img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
</p>

> A portfolio-grade NTI Machine Learning project that turns a loan-status classification experiment into a reusable ML system and an interactive product demo.

## 01 — What we built

```text
loan_data.csv
      ↓
EDA / research notebook
      ↓
Reusable ML core — src/model.py
      ├──────────────┬──────────────┐
      ↓              ↓              ↓
Prediction Studio  Batch Lab   Model Insights
      └──────────────┴──────────────┘
                     ↓
          16-slide in-app presentation
```

The repository separates research, reusable ML logic, UI, tests and documentation.

## 02 — Benchmark

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.55% | 0.64 | **0.92** | 0.75 |
| KNN | 86.29% | 0.64 | 0.88 | 0.74 |
| Decision Tree | 89.28% | 0.73 | 0.82 | 0.77 |
| Random Forest | 89.48% | 0.71 | 0.88 | 0.79 |
| SVM | 88.02% | 0.67 | **0.92** | 0.77 |
| **XGBoost** | **92.87%** | **0.87** | 0.80 | **0.83** |

XGBoost is the selected benchmark model because it leads this comparison on test accuracy, class-1 precision and class-1 F1. Logistic Regression and SVM have higher class-1 recall, so this is a trade-off rather than a universal superiority claim.

## 03 — Dataset

- `loan_data.csv` is included in the repository.
- Rows after preprocessing: **44,990**
- Processed model features: **20**
- Target: `loan_status`
- Original training distribution: **27,991 / 8,001**
- After SMOTETomek: **27,887 / 27,887**

The raw inference contract contains 13 applicant fields:

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

`loan_id` is optional for batch uploads. `loan_status` is optional and is used only to evaluate uploaded labelled data.

The notebook identifies the dataset source as Kaggle, but the exact URL is not recorded in the project files and is therefore not fabricated.

## 04 — ML pipeline

1. Remove `loan_id`.
2. IQR clipping on selected numeric variables.
3. Filter unrealistic age and employment-experience values.
4. Encode binary/ordinal variables and one-hot encode nominal variables.
5. Train/test split with `random_state=42`.
6. Fit `RobustScaler` on training data only.
7. Apply `SMOTETomek` to scaled training data only.
8. Benchmark six classifiers.
9. Select XGBoost from the benchmark.
10. Reuse the same preprocessing contract for inference.

## 05 — Product UI

### Prediction Studio

- Full raw applicant input contract
- Numeric controls bounded by ranges derived from the reference CSV
- Real XGBoost inference
- Approval probability
- Decision signals
- Benchmark context
- Session prediction history

### Batch Lab

```text
Download template
      ↓
Fill applicant rows
      ↓
Upload CSV
      ↓
Schema validation
      ↓
Batch XGBoost inference
      ↓
Inspect results
      ↓
Download scored CSV
```

If `loan_status` exists in the upload, the app also reports correctness and uploaded-label accuracy.

### Project Presentation

The app contains a **16-slide technical presentation** covering:

1. Project opening and team
2. Problem framing
3. Dataset
4. Data quality
5. Preprocessing
6. Class imbalance
7. Model benchmark
8. XGBoost selection
9. Generalization / overfitting awareness
10. Product architecture
11. Prediction Studio
12. Batch Lab
13. Model Insights
14. Limitations
15. Production roadmap
16. Team / closing

### Model Insights

- Model benchmark chart
- Full comparison table
- XGBoost feature importance
- Confusion matrix
- Precision/recall trade-off
- Generalization discussion

### Team & About

The app includes the four team members and an official NTI branding reference.

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

## 07 — Engineering design

```text
Notebook       = research / experimentation
src/           = reusable ML logic
app.py         = product + presentation layer
tests/         = quality checks
.github/       = CI automation
docs/          = architecture + model governance
```

`src/model.py` exposes a `LoanModelBundle` containing the estimator, scaler, feature schema and clipping bounds. `predict_one()` handles single-row inference and `predict_batch()` handles CSV-style batch inference using the same transformation contract.

## 08 — Run locally

```bash
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL, normally `http://localhost:8501`.

## 09 — Quality

The repository includes reusable inference functions, unit tests, GitHub Actions CI, Streamlit theme configuration, architecture documentation and a model card.

## 10 — Limitations & responsible use

This is an **educational / portfolio ML demonstration**, not a production credit-underwriting system.

Known limitations:

- The dataset may not represent a real lender's current population.
- Correlation is not a complete measure of feature usefulness.
- Class balancing changes the training distribution.
- The benchmark has not been audited for fairness, calibration or dataset shift.
- No real-time production data integration is included.
- Model probabilities are not guaranteed real-world approval odds.

**Never use this model as the sole basis for a real financial decision.**

## 11 — Production roadmap

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

## 12 — Team

**NTI Machine Learning Track**

1. **Yusuf Ayman Tolba**
2. **Mohamed Reda Hussein**
3. **Abdelrahman Mohamed Ahmed**
4. **Abdelmoniem Ibrahim Abdelmoniem**

## 13 — Official NTI

Official institute website: https://www.nti.sci.eg/

## License

Educational NTI project. Add an appropriate license before redistributing the dataset or code outside the course context.
