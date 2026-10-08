# Baselines and class-imbalance comparison (Steps 6-7)

Code: `src/churn/train.py` (`make train`). Results are logged in MLflow, experiment `telecom-churn`, one run per variant, tagged `git_sha` and `data_sha256`. View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Setup

| Item | Value |
|---|---|
| Data | Training split only (4,000 rows, 566 churners, 14.15%). The test set was not read. |
| Validation | Repeated stratified 5-fold CV, 3 repeats = 15 fits per variant, seed 42, same folds for every variant |
| Features | The shared pipeline in `src/churn/features.py` (25 features) |
| Metrics | PR-AUC (primary) and ROC-AUC; mean and standard deviation over the 15 folds; train-vs-validation gap |

## Variants

| Variant | What it is |
|---|---|
| `dummy` | Predicts the class prior. Floor: PR-AUC equals the churn rate. |
| `logreg` | Logistic regression, `class_weight="balanced"`, C chosen by an inner 3-fold CV on PR-AUC (grid in `params.yaml`) |
| `catboost` | CatBoost, 1,000 iterations, default settings, no class weights, no resampling |
| `catboost_weighted` | CatBoost with `auto_class_weights="Balanced"` (churners weigh about 6x) |
| `catboost_smotenc` | CatBoost trained on data oversampled with SMOTENC inside each CV fold (synthetic churners; `state`, `area_code` and the 0/1 flags treated as categorical) |

## Results

| Variant | CV PR-AUC | std | Train PR-AUC | Gap | CV ROC-AUC | std |
|---|---|---|---|---|---|---|
| dummy | 0.1415 | 0.0005 | 0.1415 | 0.000 | 0.5000 | 0.000 |
| logreg | 0.5347 | 0.0239 | 0.5583 | 0.024 | 0.8745 | 0.015 |
| **catboost** | **0.8994** | 0.0256 | 0.9962 | 0.097 | 0.9244 | 0.023 |
| catboost_weighted | 0.8990 | 0.0242 | 0.9997 | 0.101 | 0.9255 | 0.022 |
| catboost_smotenc | 0.8975 | 0.0221 | 0.9967 | 0.099 | 0.9260 | 0.019 |

## Read-out

- The dummy equals the churn rate, as it should.
- CatBoost beats logistic regression by +0.36 PR-AUC and +0.05 ROC-AUC, far outside the fold spread. This agrees with the earlier default-settings experiment (CatBoost about 0.887 / 0.925, logistic regression 0.465 / 0.823); logistic regression improved here through class weights and a tuned C, and the derived features helped CatBoost a little.
- The three CatBoost variants tie. PR-AUC differs by 0.0004 (weighted) and 0.0019 (SMOTENC) against a fold standard deviation of about 0.022; ROC-AUC differs by 0.001-0.002.
- Boosted models score about 0.10 PR-AUC higher on the folds they were trained on than on validation folds (train PR-AUC near 1.0). Branch 7 tuning should look at regularization.

## Decision

Carry **plain CatBoost** forward. See `docs/decisions.md`, decision 1.

## Limits

- These are CV estimates on training data, not a final performance claim. The single test-set evaluation is Step 10.
- The standard deviations are across folds. The three CatBoost variants share folds, so a paired comparison would be tighter, but it was not computed. A real difference smaller than about 0.005 PR-AUC cannot be ruled out; it would still not justify the extra complexity of weights or oversampling.
- CatBoost runs here use default hyperparameters. Tuning (branch 7) could change the ranking slightly, but not the gap to logistic regression.
