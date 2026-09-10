# Model Card — Loan Intelligence

## Model

**Primary algorithm:** XGBoost classifier  
**Task:** Binary classification of `loan_status`  
**Benchmark leader:** HistGradientBoosting  
**Deployable repository model:** XGBoost

### Final notebook benchmark

| Model | Accuracy | Precision 1 | Recall 1 | F1 1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | **93.2874%** | 0.860537 | 0.8330 | **0.846545** | **0.976527** |
| XGBoost | **92.9651%** | **0.855434** | 0.8225 | **0.838644** | 0.975957 |
| Random Forest | 91.1202% | 0.773079 | **0.8500** | 0.809717 | 0.968021 |
| Extra Trees | 89.7755% | 0.738305 | 0.8365 | 0.784341 | 0.963371 |
| Gradient Boosting | 89.3532% | 0.717809 | 0.8585 | 0.781876 | 0.962550 |
| SVM | 87.7639% | 0.671630 | 0.8795 | 0.761637 | 0.950739 |
| Decision Tree | 88.0862% | 0.703152 | 0.8030 | 0.749767 | 0.853058 |
| Logistic Regression | 86.1747% | 0.635971 | 0.8840 | 0.739749 | 0.944234 |
| KNN | 86.2192% | 0.643396 | 0.8525 | 0.733333 | 0.926571 |

## Intended use

Educational demonstration for the NTI Machine Learning track and portfolio presentation. The model is useful for explaining the mechanics of a classification workflow, feature engineering, class balancing, and comparing candidate algorithms on the supplied dataset.

## Out of scope

This model must not be treated as an autonomous credit-underwriting engine. It has not been validated for regulatory compliance, fairness, calibration, production latency, or performance on a lender's live population.

## Dataset and preprocessing

- Raw dataset: **45,000 rows × 14 columns**
- Cleaned dataset: **44,988 rows**
- Model representation: **18 features**
- Train/test split: **80/20**, stratified, `random_state=42`
- Training rows: **35,990**
- Test rows: **8,998**
- Training rows after SMOTE: **55,980**
- Original class distribution: **35,000 class 0 / 10,000 class 1**
- Training distribution before SMOTE: **27,990 class 0 / 8,000 class 1**
- Balanced training distribution after SMOTE: **27,990 / 27,990**

### Engineered features

1. `income_to_loan_ratio = person_income / (loan_amnt + 1)`
2. `emp_length_to_age_ratio = person_emp_exp / (person_age + 1)`
3. `cred_hist_to_age_ratio = cb_person_cred_hist_length / (person_age + 1)`
4. `high_risk_income_ratio = loan_percent_income * loan_int_rate`
5. `is_renting = 1` when `person_home_ownership == "RENT"`

### Training workflow

1. Remove `loan_id` when present.
2. Keep `person_age <= 80`.
3. Keep `person_emp_exp <= 50`.
4. Add the five engineered features.
5. Label-encode object columns.
6. Split into stratified 80/20 train/test sets.
7. Apply `SMOTE(random_state=42)` to the training split only.
8. Fit `StandardScaler` on the SMOTE-resampled training data.
9. Train XGBoost on the scaled training representation.

## XGBoost diagnostics

The final notebook's XGBoost benchmark gives:

- Accuracy: **92.9651%**
- Precision: **0.855434**
- Recall: **0.8225**
- F1: **0.838644**
- ROC-AUC: **0.975957**

Confusion matrix:

```text
[[6720, 278],
 [ 355, 1645]]
```

This corresponds to:

- TN = 6,720
- FP = 278
- FN = 355
- TP = 1,645
- FPR ≈ 3.98%
- FNR = 17.75%

## Top XGBoost features

The notebook's global XGBoost feature-importance chart shows the following leading features, rounded for presentation:

| Rank | Feature | Relative importance |
|---:|---|---:|
| 1 | `previous_loan_defaults_on_file` | ~88.0% |
| 2 | `high_risk_income_ratio` | ~1.9% |
| 3 | `cb_person_cred_hist_length` | ~1.4% |
| 4 | `loan_int_rate` | ~1.0% |
| 5 | `person_home_ownership` | ~0.8% |
| 6 | `person_education` | ~0.7% |
| 7 | `loan_intent` | ~0.7% |
| 8 | `person_gender` | ~0.6% |

Feature importance is model attribution, not a causal explanation.

## Hyperparameter optimization experiment

The notebook also runs a separate `RandomizedSearchCV` experiment over XGBoost:

```text
n_estimators       = 250
max_depth           = 12
learning_rate       = 0.01
subsample           = 0.60
colsample_bytree    = 0.80
gamma               = 0.00
```

The notebook reports the resulting test classification report at approximately:

- Accuracy: **91%**
- Precision (class 1): **0.77**
- Recall (class 1): **0.86**
- F1 (class 1): **0.81**

This experiment is retained as a documented research result and is not silently promoted over the stronger benchmark XGBoost result.

## Limitations

1. The dataset may not represent current real-world applicants.
2. The benchmark is a fixed experimental evaluation, not a production validation study.
3. The categorical encoding is learned from the supplied dataset.
4. SMOTE changes the training distribution.
5. The model has not been audited for subgroup fairness.
6. The model has not been calibrated for reliable real-world probability estimates.
7. The benchmark does not establish causal relationships.
8. No model-drift monitoring or retraining policy exists yet.
9. The feature-importance chart should not be interpreted as a causal ranking of creditworthiness.
