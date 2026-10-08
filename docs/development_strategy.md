# Development Strategy: Telecom Customer Churn Prediction, Orange/BigML data (Production-Ready)

## 1. Objective and Success Criteria
- **Goal:** predict customers likely to churn in the next period so retention can intervene.
- **Dataset:** Orange/BigML telecom churn (OpenML ID 40701, name `churn`): 5,000 customers, 19 usable predictors, binary target `class` (1 = churned, 14.1%). Full profile in Section 1.1.
- **ML metrics:** PR-AUC (primary, imbalanced), ROC-AUC, recall at fixed precision, F1, calibration (Brier score).
- **Business metric:** expected retention value = (saved CLV x acceptance rate) - offer cost. Used to choose the decision threshold. The dataset has no customer-value column, so CLV, acceptance rate and offer cost are assumptions in `configs/params.yaml` (for example monthly revenue = sum of the four `total_*_charge` columns, CLV = monthly revenue x assumed remaining months) and are reported as assumptions.
- **Targets (initial):** ROC-AUC >= 0.92, PR-AUC >= 0.85, recall >= 0.75 at the cost-optimal threshold, inference p95 < 100 ms. Set from the default-settings experiment in Section 1.2 (ROC-AUC about 0.925, PR-AUC about 0.886); tuning may add little.

### 1.1 Dataset properties (profiled 2026-10-07)
| Property | Value |
|---|---|
| Source | Orange/BigML telecom churn, the widely used "churn" benchmark (MLC++ origin). OpenML 40701 holds 5,000 rows. Its origin is loosely documented and it is widely believed to be at least partly simulated, so results show the method, not real-world performance |
| Size | 5,000 rows (customers), 20 columns plus the target; small, so use repeated CV and report confidence intervals |
| Structure | One row per customer: a **snapshot with no date or time column** (no temporal split, no time-series model) |
| Target | `class`: 707 churned (1) / 4,293 stayed (0), **14.1% positive**, a stronger imbalance than Telco; a no-skill model scores PR-AUC 0.141 |
| Identifier | `phone_number` (unique 0-4,999, behaves like a row index): dropped, never a feature |
| Categorical features (4) | `state` (51 levels, stored as integer codes 0-50 in the OpenML copy), `area_code` (408 / 415 / 510), `international_plan` (0/1), `voice_mail_plan` (0/1) |
| Numeric features (15) | `account_length` (1-243, unit not documented), `number_vmail_messages` (0-52), `number_customer_service_calls` (0-9, a small count), and for day / evening / night / international: `total_*_minutes`, `total_*_calls`, `total_*_charge` (4 periods x 3 fields = 12) |
| Missing values | None |
| Duplicates | None (rows or phone numbers) |
| Redundancy | In each period `total_*_charge` is perfectly correlated with `total_*_minutes` (r = 1.00, a fixed rate per minute); `number_vmail_messages` > 0 matches `voice_mail_plan` = 1 in all but one row |
| Distributions | Usage minutes and calls are roughly bell-shaped; `number_vmail_messages` is zero-inflated (about 73% zeros); customer-service calls rise in meaning above 3 |
| Leakage | No outcome-derived columns |
| Not available | Tenure in dated terms, contract type, billing history, customer value, demographics (age, gender), timestamps |

**What this implies for modelling**
- All predictors are already numeric or low-to-medium cardinality, so the work is mostly feature engineering and the `state` encoding, not cleaning.
- The signal is non-linear and interaction-driven (for example heavy daytime use together with no voicemail, or four or more service calls), which is why a linear model underperforms (Section 1.2).
- 707 churners is a small positive class. Hold-outs of 15% would contain only about 106 churners each, so use an 80/20 split with repeated CV on the training part (Step 4).
- No protected attributes exist, so a protected-group fairness audit is not possible; use segment error analysis instead (Step 10).

### 1.2 Model experiment on this dataset (default parameters, stratified 5-fold CV, same folds)
| Model | PR-AUC | ROC-AUC | Time (5 folds) |
|---|---|---|---|
| Logistic regression (impute, one-hot, scaling) | 0.465 | 0.823 | 0.3 s |
| LightGBM | 0.886 | 0.926 | 0.8 s |
| XGBoost | 0.881 | 0.925 | 2.8 s |
| **CatBoost** | **0.887** | 0.925 | 29 s |

No-skill PR-AUC is 0.141. Fold-to-fold spread is about 0.02 PR-AUC.
- All three boosted models beat logistic regression by a very large margin (about +0.42 PR-AUC and +0.10 ROC-AUC), far beyond the noise. This is the dataset where boosting is justified.
- The three boosters are a statistical tie with each other. **Decision:** CatBoost is the single boosted model (project decision; native handling of the 51-level `state`, simple train/serve pipeline). It trains much slower than LightGBM for the same accuracy, which is accepted and noted in the model card.
- Reproduce with `experiments/boosting_vs_logreg.py 40701 orange_bigml phone_number state,area_code,international_plan,voice_mail_plan`. The same script on the 7,043-row Telco file showed no boosting advantage (logistic regression and CatBoost tied), which is why this dataset was chosen.

## 2. Tech Stack
| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Data/EDA | pandas, numpy, matplotlib, seaborn, ydata-profiling |
| Validation | pandera or Great Expectations |
| Modeling | scikit-learn (pipelines, logistic regression baseline), CatBoost (only boosted model), imbalanced-learn |
| Tuning | Optuna |
| Explainability | SHAP |
| Calibration / segment analysis | sklearn CalibratedClassifierCV, Fairlearn (segment metrics only) |
| Experiment tracking / registry | MLflow |
| Data/model versioning | MLflow (DVC dropped for v1) |
| Serving | FastAPI + Uvicorn, Pydantic schemas |
| Batch scoring | scheduled script/Prefect |
| Packaging | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest, pytest-cov |
| Code quality | ruff, black, mypy, pre-commit |
| Monitoring | Evidently (drift), Prometheus + Grafana (service metrics) |
| Config | YAML + pydantic-settings |

## 3. Repository Structure
```
TelecomChurnPrediction/
├── data/ (raw, interim, processed)   # git-ignored
├── notebooks/                         # EDA only
├── src/churn/ (data, features, train, evaluate, predict, monitor)
├── api/ (main.py, schemas.py)
├── configs/                           # params.yaml
├── tests/
├── models/                            # MLflow artifacts
├── Dockerfile, pyproject.toml, Makefile
└── .github/workflows/
```

## 4. Development Steps

### Step 1: Problem framing and data audit
- Define churn (`class` = 1), prediction horizon (not stated in the source; assume "next billing period"), and the retention action (offer or call for flagged customers).
- Write a data contract: column types, allowed values, ranges. Document that `state` codes are anonymous integers and `account_length` has no stated unit.

### Step 2: Data ingestion and validation
- Load the raw OpenML/CSV file once and keep it in `data/raw/` unchanged; never mutate raw data.
- Cleaning: drop `phone_number` (identifier); cast `state`, `area_code` to categorical strings; keep `international_plan` and `voice_mail_plan` as 0/1; keep `number_customer_service_calls` as an integer count.
- Schema checks (pandera): `area_code` in {408, 415, 510}, `state` in 0-50, plan flags in {0,1}, `number_customer_service_calls` 0-9, minutes, calls and charges non-negative, `account_length` >= 1, target in {0,1}, no missing values, no duplicate rows.
- **Leakage check:** no outcome-derived columns exist. The check fails the run if any identifier-like column (unique per row) remains in the features, or if any single feature separates the target perfectly. Document each drop.

### Step 3: EDA
**Rule:** EDA that looks at the target runs on the **training split only**, so the held-out test set never influences decisions. Do Step 4 (split) first, then run EDA on `train`. Only structure and data-quality checks (3.1) use the full file. The patterns quoted below come from the initial profile and are hypotheses to re-verify on the training split.

**3.1 Data quality (full file)**
- Shape (5,000 rows), dtypes (the OpenML copy stores categoricals as integers, so decode them), missing values (expect none), duplicate rows, unique `phone_number`.
- Ranges: non-negative minutes, calls and charges; `number_customer_service_calls` 0-9; `account_length` >= 1.
- Consistency: `voice_mail_plan` = 1 exactly when `number_vmail_messages` > 0 (one known exception); charge = rate x minutes in every period.

**3.2 Univariate (train)**
- Histograms and boxplots for the 12 usage fields (minutes, calls, charges across day/evening/night/international), `account_length`, `number_vmail_messages` (zero-inflated).
- Count plots for `number_customer_service_calls`, `state` (about 100 customers per state, so flag small groups), `area_code`, `international_plan`, `voice_mail_plan`.

**3.3 Target analysis (train)**
- Class balance (about 14.1% churn) and the no-skill PR-AUC baseline (= churn rate).
- Churn rate by each categorical column with 95% confidence intervals and chi-square tests. Expected: international-plan holders churn at roughly 42% versus about 11% without the plan.
- Churn rate by `number_customer_service_calls` (expected: about 11% for 0-3 calls, then a jump to 44-65% from 4 calls up; the 9-call group is tiny).
- Distributions of every numeric feature by churn class (KDE/boxplots), including `total_day_minutes` (heavy users) and `total_intl_minutes`.

**3.4 Relationships and redundancy (train)**
- Correlation matrix and variance-inflation factors; confirm each `total_*_charge` duplicates its `total_*_minutes` (r = 1.00) so one of each pair can be dropped for linear models.
- Mutual information of every feature with the target as a first importance ranking, plus a quick look for interactions (for example day minutes x international plan).
- Cramer's V between the categorical columns.

**3.5 Segments (train)**
- Churn by `international_plan` x `voice_mail_plan`, service calls bucket (0-3 vs 4+) x `international_plan`, day-usage tertile, `area_code`, and the highest- and lowest-churn `state` codes (with confidence intervals and a small-sample warning).

**3.6 Outputs**
- `notebooks/01_eda.ipynb` (plots and commentary) and a ydata-profiling HTML report on the training split in `reports/eda/`.
- A short findings list that feeds Step 2 (cleaning), Step 5 (which charge columns to drop, which derived features to build) and Step 6 (imbalance handling).
- Tools: pandas, matplotlib, seaborn, scipy, ydata-profiling. No logic that must run in production lives in the notebook.

### Step 4: Train/validation/test split
- Stratified **80/20** (about 141 churners in the test set) with repeated stratified 5-fold CV on the 80% for tuning, calibration and threshold selection. A 70/15/15 split would leave only about 106 churners in each hold-out, which is too few. Fixed seed.
- Test set locked until final evaluation. All fitting done inside CV folds to avoid leakage.

### Step 5: Feature engineering (in sklearn Pipeline/ColumnTransformer)
- Categorical: `state` (51 levels) and `area_code` go to CatBoost directly through `cat_features`. For the logistic-regression baseline, one-hot encode them, grouping rare levels.
- Binary: `international_plan`, `voice_mail_plan` stay 0/1.
- Redundancy: drop the four `total_*_charge` columns from the shared feature set after computing `total_charge` (they duplicate minutes at a fixed rate); confirm in EDA.
- Derived features: `total_minutes`, `total_calls`, `total_charge` (monthly-revenue proxy), average call length per period (minutes / calls), day-share of minutes, international minutes per call, `high_service_calls` (>= 4), `has_voicemail_usage` (`number_vmail_messages` > 0), `account_length` buckets.
- Numeric: StandardScaler for the linear model only (not needed for CatBoost).
- Encapsulate in one fitted pipeline so training and serving use identical transforms.

### Step 6: Handling class imbalance
- Churn is 14.1%. Primary: `auto_class_weights` / `scale_pos_weight` (preferred, no data distortion).
- Compare against SMOTE/SMOTENC applied **inside CV only**.
- Evaluate with PR-AUC; tune the threshold rather than relying on 0.5.

### Step 7: Baselines and primary model
**Primary model: CatBoost, the only boosted model.** In the Section 1.2 experiment all three boosters tied (PR-AUC about 0.88) and beat logistic regression by a wide margin. CatBoost is kept for native categorical handling of `state`, ordered boosting and strong default regularization on small data, and a single-model train/serve pipeline. XGBoost and LightGBM are not carried forward.
1. Dummy classifier (floor, PR-AUC = churn rate).
2. Logistic Regression (L1/L2, tuned `C`): the interpretable baseline. It is expected to trail CatBoost clearly (default settings gave PR-AUC 0.465).
3. **CatBoost (primary).**
- Compare via stratified 5-fold repeated CV; log each run in MLflow.
- CatBoost receives categorical columns directly via `cat_features`; the Logistic Regression baseline uses the one-hot pipeline from Step 5.
- Report the gap with confidence intervals, and note that LightGBM matches CatBoost at a fraction of the training time.

### Step 8: Hyperparameter tuning
- Optuna (TPE) with stratified CV, objective = PR-AUC, with pruning; budget 100-200 trials on CatBoost (`depth`, `learning_rate`, `l2_leaf_reg`, `iterations` with early stopping, `auto_class_weights`).
- Log params, metrics, and artifacts to MLflow.

### Step 9: Calibration and threshold selection
- Calibrate probabilities (isotonic/Platt) on a held-out fold; check reliability curve.
- Choose threshold on validation set by maximizing expected business value (cost matrix: offer cost vs. lost CLV), not F1 alone.

### Step 10: Final evaluation
- One-time run on the test set: ROC-AUC, PR-AUC, recall/precision at threshold, confusion matrix, lift/gain charts, calibration.
- Error analysis by segment (`international_plan`, service-call bucket, `voice_mail_plan`, `area_code`, high- and low-churn `state` groups). The dataset has no protected attributes (age, gender), so a protected-group fairness audit is not possible; document this as a limitation. Fairlearn is only used for the segment comparison.
- Bootstrap confidence intervals.

### Step 11: Explainability
- Global: SHAP summary, feature importance (column names are descriptive, so reasons read clearly, for example "high `total_day_minutes`"; `state` is only an anonymous code).
- Local: per-customer SHAP reasons returned with each prediction (feeds retention playbook).

### Step 12: Productionization
- Serialize full pipeline + model + threshold + metadata (versioned) via MLflow Model Registry (Staging -> Production).
- **FastAPI service:** `POST /predict` (single), `POST /predict/batch`, `GET /health`, `GET /model-info`. Pydantic validates inputs against the data contract.
- Dockerize (slim image, non-root, pinned dependencies); batch scoring job for nightly runs.
- Latency and load test (Locust).

### Step 13: Testing
- Unit: feature transforms, schema validators, threshold logic.
- Integration: end-to-end pipeline on a sample; API contract tests.
- Model tests: minimum metric gate, invariance/directional tests (more customer-service calls or having an international plan should not lower predicted risk, all else equal), determinism with a seed.

### Step 14: CI/CD
- GitHub Actions: lint -> type check -> tests -> train on small sample -> build image -> push -> deploy to staging -> smoke test -> manual approve -> prod.
- Model promotion gated on beating the current champion on the metric and passing segment-performance checks.

### Step 15: Monitoring and retraining
- Data drift (PSI/KS via Evidently), prediction drift, and service metrics (latency, error rate).
- Performance monitoring once ground-truth labels arrive (delayed feedback loop).
- Retrain triggers: scheduled (e.g., monthly) or drift/performance-decay alert; champion/challenger or shadow deployment before promotion.
- Alerts to Slack/email; rollback to the previous registry version.

### Step 16: Documentation and governance
- Model card (intended use, data, metrics, limits, segment analysis, dataset caveats), data dictionary, runbook, ADRs.
- PII handling: `phone_number` is an identifier and is never a feature or logged; access control on the registry.

## 5. Milestones
| Phase | Deliverable |
|---|---|
| 1 | Repo scaffold, data validation, EDA |
| 2 | Feature pipeline, baselines, MLflow tracking |
| 3 | Tuned and calibrated model, final evaluation, SHAP |
| 4 | API, Docker, tests, CI/CD |
| 5 | Monitoring, model card, runbook |

## 6. Risks and Mitigations
- **Small dataset (5k rows, 707 churners):** use repeated CV, regularization, keep hold-outs large enough (80/20); report CIs.
- **Origin and realism:** the dataset is old, loosely documented and widely believed to be at least partly simulated; high scores (ROC-AUC about 0.92) may not carry over to real operator data. State this in the model card.
- **Leakage:** `phone_number` is an identifier and is dropped; the leakage check fails the run on identifier-like or perfectly separating features.
- **Anonymous `state` codes and unspecified `account_length` unit:** treat as categorical and ordinal respectively; do not interpret them.
- **No protected attributes, no customer value:** no protected-group fairness audit; the business threshold relies on documented assumptions.
- **Static snapshot dataset:** no temporal split possible; note the limitation and design the pipeline for time-based splits on real data.
- **Overfitting by tuning:** nested/held-out test, single final test pass.
- **Drift in production:** monitoring and retrain triggers above.
