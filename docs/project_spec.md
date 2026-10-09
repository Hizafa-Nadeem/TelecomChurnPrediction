# Project Spec: Telecom Customer Churn Prediction (Orange/BigML data)

Source documents: `project_scope.md` (scope), `development_strategy.md` (Steps 1-12 define the work), `Azure_production_plan.md` (Phase 2 target).

## 1. Scope

> "I want a local project first then a demo on Azure." (`project_scope.md`)

| Phase | Goal | Cost |
|---|---|---|
| **Phase 1: Local project** | Build, train, evaluate, explain, package and serve the churn model entirely on the local machine. | Free, no Azure account needed |
| **Phase 2: Azure demo** | Deploy the Phase 1 project to Azure ML as a short live demo (train pipeline, registered model, online endpoint), then tear it down. | Azure for Students / trial credit, deleted after the demo |
| **Phase 3 (v3): Production hardening** | **Only if time permits.** Batch scoring, monitoring and CI/CD (section 11). | Local/free-tier first; Azure only if credit remains |

**In scope (v1 and v2):** Steps 1-12 of `development_strategy.md` (problem framing through productionization).

**Deferred to v3, if time permits:** batch processing, monitoring and CI/CD pipelines (section 11), plus the related development-strategy steps 13-15. Governance docs (Step 16) are also v3. A few smoke tests are still needed in v1 so the model can be checked before the demo. v3 is a stretch goal and is not required for v1 or v2 to be complete.

**Key requirement:** the tech stack and project structure must be reusable for the Azure demo without a rewrite. See section 4.

## 2. Problem and Success Criteria

- **Goal:** predict which customers will churn so retention can intervene.
- **Dataset:** Orange/BigML telecom churn (OpenML ID 40701, name `churn`): 5,000 customers, 19 usable predictors (15 numeric, 4 categorical: `state` with 51 levels, `area_code`, `international_plan`, `voice_mail_plan`), binary target `class` (1 = churned, 707 / 14.1%). `phone_number` is an identifier and is dropped. Properties: no missing values, no duplicates, no time column (snapshot), no protected attributes, no customer-value column, each `total_*_charge` duplicates its `total_*_minutes`. The data is old, loosely documented and probably partly simulated, so results show the method, not real-world performance. Full profile in `development_strategy.md` Section 1.1.
- **Primary metric:** PR-AUC. Also report ROC-AUC, recall at fixed precision, F1 and Brier score.
- **Business metric:** expected retention value = (saved CLV x acceptance rate) - offer cost. It is used to choose the decision threshold. The dataset has no customer-value column, so CLV, acceptance rate and offer cost are assumptions in `configs/params.yaml` (for example monthly revenue = sum of the four `total_*_charge` columns) and are reported as assumptions.
- **Initial targets:** ROC-AUC >= 0.92, PR-AUC >= 0.85, recall >= 0.75 at the cost-optimal threshold, inference p95 < 100 ms. Set from a default-settings experiment (CatBoost PR-AUC 0.887, ROC-AUC 0.925); see `development_strategy.md` Section 1.2.

## 3. Phase 1 Deliverables (Steps 1-12)

| Step | Deliverable | Output |
|---|---|---|
| 1 | Problem framing: churn definition, horizon, retention action, data contract | `docs/data_contract.md`, pandera schema |
| 2 | Ingestion and validation: drop `phone_number` (identifier), cast `state` and `area_code` to categorical, pandera schema checks (ranges, allowed values, no missing, no duplicates); leakage check fails the run if an identifier-like or perfectly separating feature remains; each drop documented | `data/interim/`, validation report |
| 3 | EDA on the **training split only** (split first): data quality, univariate (12 usage fields, zero-inflated voicemail messages, service calls, `state`), churn rate by feature with confidence intervals and chi-square tests (international plan, service-call count), charge-vs-minutes redundancy, correlation, mutual information, segments (international plan x voicemail, service calls x international plan, day-usage tertile, `state` groups). Details in `development_strategy.md` Step 3 | `notebooks/01_eda.ipynb`, `reports/eda/` profiling report, findings list (no production logic) |
| 4 | Stratified 80/20 train/test split (about 141 churners in the test set), repeated 5-fold CV on the training part for tuning, calibration and threshold, fixed seed, test set locked until Step 10 | `data/processed/` |
| 5 | Feature engineering in one fitted pipeline (shared by training and serving): categorical `state`/`area_code`, drop the redundant charge columns after computing `total_charge`, derived totals, average call length, `high_service_calls`, voicemail usage | `src/churn/features.py` |
| 6 | Class imbalance handling: plain, class-weighted and SMOTENC-in-CV CatBoost compared; all tied, so plain CatBoost is used (`reports/baselines.md`) | experiment logged in MLflow |
| 7 | Baselines (dummy, tuned logistic regression) and **CatBoost as the only boosted model**. A quick experiment (5-fold CV, default settings) gave PR-AUC 0.465 / ROC-AUC 0.823 for logistic regression versus 0.887 / 0.925 for CatBoost, 0.886 / 0.926 for LightGBM and 0.881 / 0.925 for XGBoost. Boosting clearly beats the linear model, and the three boosters tie. CatBoost is kept for native handling of the 51-level `state` and a single-model pipeline; XGBoost and LightGBM are not carried forward | MLflow runs |
| 8 | Optuna tuning (PR-AUC, stratified CV, pruning, 30 trials; flat landscape, see `reports/tuning.md`) | best params, MLflow runs |
| 9 | Calibration (none/Platt/isotonic, best by Brier) and business-cost threshold selection (`reports/calibration.md`) | calibrator, threshold, reliability curve |
| 10 | One-time test-set evaluation: metrics, confusion matrix, lift/gain, bootstrap CIs, segment error analysis (international plan, service-call bucket, voicemail plan, area code, `state` groups). No protected attributes exist, so a protected-group fairness audit is not possible and is documented as a limitation | `reports/final_evaluation.md` |
| 11 | Explainability: global SHAP and per-customer top reasons | SHAP plots, reason logic in the model |
| 12 | Productionization: MLflow pyfunc model (pipeline + CatBoost + calibrator + threshold + SHAP reasons), registered; FastAPI service (`/predict`, `/predict/batch`, `/health`, `/model-info`); Dockerfile; Locust load test | registered model, `api/`, `Dockerfile` |

## 4. Azure-Ready Design Rules

These rules apply from the first line of code, so Phase 2 is a deployment task and not a rewrite.

| Rule | Why (Azure mapping) |
|---|---|
| Each stage is a Python module with a CLI (`python -m churn.<stage> --config ...`) that reads inputs from disk and writes outputs to disk | Each stage becomes one Azure ML pipeline component with no code change |
| All parameters live in `configs/params.yaml` (plain `key: value`, 2-space indent), loaded with `yaml.safe_load` and validated by Pydantic models in `src/churn/config.py`; paths are passed as arguments and never hard-coded | AML jobs inject data and output paths as arguments |
| Track with **MLflow** from day one (local `mlruns/` or `sqlite`) | The AML workspace is MLflow-compatible, so only the tracking URI changes |
| Package the model as one **MLflow pyfunc** model; `predict()` returns a flat table (`churn_probability`, `decision`, `reason_1..3`) | Deploys as a managed online endpoint with no scoring script |
| Pin the environment (`pyproject.toml` plus a lock file, and a `conda.yaml` for the model) | AML environments and the endpoint image are built from it |
| Pydantic schemas mirror the data contract | Same validation locally and in the API/endpoint |
| Docker image for the FastAPI service | Fallback option: Azure Container Apps |
| Local data folders mirror Blob layout (`raw/`, `interim/`, `processed/`) | They map to Azure ML Data Assets |
| No secrets in code; settings come from environment variables via pydantic-settings | Replaced by Key Vault/managed identity on Azure |

## 5. Tech Stack

Stack choices are from `development_strategy.md` Section 2, limited to Steps 1-12.

| Layer | Local (Phase 1) | Azure equivalent (Phase 2) |
|---|---|---|
| Language | Python 3.11 | same |
| Data/EDA | pandas, numpy, matplotlib, seaborn, ydata-profiling | same |
| Validation | pandera | first pipeline component |
| Modeling | scikit-learn, CatBoost (only boosted model), imbalanced-learn (SMOTE comparison in Step 6) | same, on AML compute cluster |
| Tuning | Optuna | same, as an AML pipeline job |
| Explainability | SHAP | same, inside the pyfunc model |
| Calibration / segment analysis | CalibratedClassifierCV, Fairlearn (segment metrics only) | same |
| Tracking/registry | MLflow (local) | Azure ML workspace MLflow and model registry |
| Pipeline/versioning | Makefile stages (DVC dropped for v1) | AML pipeline job and Data Assets |
| Serving | FastAPI + Uvicorn, Docker | Managed online endpoint (or Container Apps) |
| Config | YAML file + Pydantic validation (pydantic-settings for env vars) | same |
| Quality | ruff, black, mypy, pre-commit | same |
| Load test | Locust | same |

## 6. Repository Structure

```
TelecomChurnPrediction/
├── data/ (raw, interim, processed)     # git-ignored, raw is never edited
├── notebooks/                          # EDA only
├── src/churn/                          # data, features, train, tune, calibrate, evaluate, explain, predict, pyfunc_model
├── api/                                # main.py, schemas.py (FastAPI)
├── configs/                            # params.yaml, train.yaml
├── reports/                            # evaluation, SHAP, segment analysis
├── docs/                               # project_scope, project_spec, development_strategy, Azure_production_plan, data_contract
├── tests/                              # smoke tests (full suite is Step 13, later)
├── azure/                              # Phase 2 only: Bicep, AML job/endpoint YAML (empty in Phase 1)
└── Dockerfile, pyproject.toml, Makefile
```

## 7. Phase 2: Azure Demo

Uses a **subset** of `Azure_production_plan.md`. It is a short demo and not a full production setup.

1. One resource group and an AML workspace (`dev` only), using Bicep or the CLI. Skip the managed VNet and private endpoints.
2. Register the raw CSV as a Data Asset.
3. Run the training pipeline as an AML pipeline job (CPU cluster, min nodes 0).
4. Register the MLflow pyfunc model.
5. Deploy a managed online endpoint (1 instance) as `blue` and send test requests.
6. **Mandatory: blue/green safe rollout.** Deploy a second model version as `green` with 0% traffic and test it directly. Mirror a share of traffic to it. Split traffic (`blue=90 green=10`), cut over to `green=100`, then roll back to `blue=100`. Capture the traffic settings and the responses at each stage.
7. Capture screenshots and logs, then delete the resource group.

**Not in the demo (moved to v3, see section 11):** batch endpoints, model monitoring, retraining schedules, CI/CD with OIDC, and the cross-workspace registry.

## 8. Run Practice

Every stage has a `make` target, and the stages run in this order: `make setup` -> `make data` -> `make train` -> `make evaluate` -> `make serve`. A failed data contract stops the run. Only a model that passes the promotion gate (beats the logistic-regression baseline on PR-AUC and passes the calibration and segment-performance checks) is registered. Every run logs its config, git SHA and data hash to MLflow.

## 9. Acceptance Criteria

**Phase 1 is done when:**
- [ ] `make all` runs from a clean checkout and reproduces the same metrics with the same seed.
- [ ] The test set is used once, and the final report has bootstrap CIs.
- [ ] The model meets the initial targets (section 2), or the report explains the gap.
- [ ] The registered pyfunc model gives the same output as the local pipeline on a fixture set.
- [ ] The API returns probability, decision and top 3 reasons, and runs in Docker.
- [ ] Locust shows p95 latency < 100 ms.
- [ ] No leaked columns are used (`phone_number` dropped; the leakage check passes).

**Phase 2 is done when:**
- [ ] The same stage modules and configs run as an AML pipeline job without code changes.
- [ ] The registered model is served by a live managed online endpoint and returns the same output format.
- [ ] Blue/green rollout is exercised end to end: green at 0% tested directly, mirroring, a 90/10 split, full cutover, and rollback to blue.
- [ ] The Azure resources are deleted after the demo.

## 10. Risks (v1 and v2)

- **Small dataset (5k rows, 707 churners):** repeated CV, regularization, an 80/20 split so the test set holds enough churners, report confidence intervals.
- **Leakage:** `phone_number` is an identifier and is dropped; a leakage check fails the run if an identifier-like or perfectly separating feature remains.
- **Origin and realism:** the dataset is old, loosely documented and probably partly simulated; the high scores may not carry over to real operator data. State this in the model card.
- **No protected attributes and no customer value:** no protected-group fairness audit; the business threshold relies on documented assumptions.
- **Anonymous `state` codes and unspecified `account_length` unit:** treat as categorical and ordinal; do not interpret them.
- **Static snapshot, no temporal split:** state the limitation and keep the pipeline ready for time-based splits.
- **Overfitting through tuning:** held-out test set, a single final pass.
- **Azure cost:** keep compute at min nodes 0, use one endpoint instance, and delete resources right after the demo.
- **Phase 2 rewrite risk:** the rules in section 4 are the mitigation. Break them and the demo costs extra time.
- **Azure SDK v1 is end of support:** use CLI/SDK v2 and MLflow only.

## 11. Phase 3 (v3): Batch Processing, Monitoring, CI/CD (if time permits)

Start v3 only after v1 and v2 meet their acceptance criteria. Do the items in this order and stop whenever time runs out. Each item is useful on its own.

| Priority | Item | Purpose | Local / free version | Azure version (if credit remains) |
|---|---|---|---|---|
| 1 | **CI/CD pipeline** (dev strategy Steps 13-14) | Test and ship changes automatically and safely | GitHub Actions: lint, mypy, pytest, train on a small sample, build the Docker image, model quality gate | GitHub Actions with `azure/login` (OIDC), `az ml job create`, staging then manual approval then prod |
| 2 | **Batch processing** | Score the whole customer base on a schedule | A `churn.batch_predict` CLI scoring a CSV/Parquet snapshot and writing a unique output file; scheduled with GitHub Actions cron or Prefect | Batch endpoint on the same MLflow model, output to a blob datastore |
| 3 | **Monitoring** (dev strategy Step 15) | Detect drift and decay after deployment | Evidently drift report (PSI/KS) comparing new data to training data; latency/error metrics from the FastAPI service | AML data collection and model monitoring schedule, with Azure Monitor alerts |
| 4 | **Retraining loop** | Act on what monitoring finds | Scheduled workflow or manual `make train` with the promotion gate | AML schedule on the training pipeline |
| 5 | **Governance** (dev strategy Step 16) | Document and control the model | Model card, data dictionary, runbook, ADRs | Same documents, plus RBAC and Key Vault |

**Already prepared in v1:** the design rules in section 4 (CLI stages, config files, MLflow, pyfunc model, pinned environments) are what make these add-ons cheap to build later.

**v3 acceptance criteria (per item, all optional):**
- [ ] CI runs lint, tests and a small training job on every push and fails on a quality drop.
- [ ] The batch command scores a snapshot and writes a unique output file, run on a schedule.
- [ ] A drift report is generated on a new data sample and flags an injected shift.
- [ ] A model card and runbook exist in the repo.

**v3 risks:** time (it is a stretch goal, so drop items instead of rushing them) and Azure cost (keep v3 on the local/free versions unless credit remains).
