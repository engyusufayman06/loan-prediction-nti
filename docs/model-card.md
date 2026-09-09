# Model Card — Loan Intelligence

## Model

**Algorithm:** XGBoost classifier  
**Task:** Binary classification of `loan_status`  
**Benchmark test accuracy:** 92.87%

## Intended use

Educational demonstration for the NTI Machine Learning track and portfolio presentation. The model is useful for explaining the mechanics of a classification workflow and comparing candidate algorithms on the supplied dataset.

## Out of scope

This model must not be treated as an autonomous credit-underwriting engine. It has not been validated for regulatory compliance, fairness, calibration, production latency, or performance on a lender's live population.

## Training workflow

- Remove identifier: `loan_id`
- IQR clipping on selected numeric features
- Filter `person_age <= 80`
- Filter `person_emp_exp <= 60`
- Encode categorical variables
- 80/20 train-test split with `random_state=42`
- `RobustScaler` fitted on the training split
- `SMOTETomek` applied to the scaled training data
- XGBoost fitted on the balanced training data

## Benchmark

| Model | Accuracy | Precision 1 | Recall 1 | F1 1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.55% | 0.64 | 0.92 | 0.75 |
| KNN | 86.29% | 0.64 | 0.88 | 0.74 |
| Decision Tree | 89.28% | 0.73 | 0.82 | 0.77 |
| Random Forest | 89.48% | 0.71 | 0.88 | 0.79 |
| SVM | 88.02% | 0.67 | 0.92 | 0.77 |
| XGBoost | **92.87%** | **0.87** | 0.80 | **0.83** |

## Limitations

1. The dataset may not represent current real-world applicants.
2. The original notebook's preprocessing includes IQR clipping before the train/test split; this is retained for consistency with the submitted benchmark.
3. Accuracy alone does not capture the cost of false approvals versus false rejections.
4. The model has not been audited for subgroup fairness.
5. The model has not been calibrated for reliable probability estimates.
6. No model-drift monitoring or retraining policy exists yet.
