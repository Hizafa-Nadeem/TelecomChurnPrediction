# Azure Production Plan: Telecom Churn Prediction (Orange/BigML data, CatBoost)

Companion to `development_strategy.md`. It covers how to take the trained CatBoost pipeline and run it in production on Azure.

> **Documentation basis:** reviewed against Microsoft Learn on 2026-10-06 (Azure Machine Learning CLI/SDK **v2**). Statements backed by docs carry a [S#] tag that maps to the Sources table at the end. Items marked **(design choice)** are recommendations of this plan, not documented requirements. Some Azure ML features are labeled *preview* in the docs; they are flagged where used.

## 1. Architecture Overview

```
                    ┌─────────────────────────── GitHub Actions (CI/CD, azure/login via OIDC) ───────────────────────────┐
                    │ lint/test -> run AML training pipeline -> register model -> deploy staging -> approve -> prod       │
                    └──────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                                   │
 Raw data ──> Blob / ADLS Gen2 ──> Azure ML pipeline job (validate, train CatBoost, tune, calibrate, evaluate)
 (Data Assets)   datastore              │ tracked with MLflow in the workspace
                                        ▼
                         Azure ML model registry (workspace dev)  ──share──>  Azure ML registry (cross-workspace)
                                                                                      │
                          ┌───────────────────────────────────────────────────────────┴──────────────────────┐
                          ▼                                                                                  ▼
        Managed Online Endpoint (aad_token auth)                                           Batch Endpoint (nightly scoring)
        blue / green deployments, traffic split, mirror                                     output -> blob datastore
        Data collector -> model_inputs / model_outputs                                              │
                          │                                                                         ▼
                          ▼                                                           Prediction store (Azure SQL / Parquet)
        Azure ML Model Monitoring (schedule): data drift, prediction drift, data quality
                          │ threshold breach -> email / Event Grid -> retraining trigger
                          ▼
        Azure Monitor + Log Analytics + Application Insights (latency, errors, logs, alerts)

 Cross-cutting: Key Vault · Managed Identity · ACR · workspace managed VNet + private endpoints · Bicep/Terraform · Azure Policy
```

## 2. Azure Tech Stack

| Layer | Azure service / tool | Purpose | Notes |
|---|---|---|---|
| Source control | GitHub (or Azure Repos) | Code, configs, workflows | |
| CI/CD | GitHub Actions (or Azure DevOps Pipelines) | Test, train, deploy | OIDC via `azure/login` [S8]; MLOps v2 reference [S9] |
| Data storage | Azure Blob / ADLS Gen2 | Raw and processed data, outputs | Batch outputs must land in a **blob** datastore [S5] |
| Data versioning | Azure ML Data Assets | Versioned datasets | DVC is optional once Data Assets are used (design choice) |
| ML workspace | Azure Machine Learning (SDK/CLI **v2**) | Jobs, models, endpoints, monitoring | SDK v1 is end of support (see Risks) [S5] |
| Training compute | AML compute cluster (CPU, min nodes 0) | Training and Optuna jobs | |
| Pipelines | AML pipeline jobs + components | `validate -> split -> train -> tune -> calibrate -> evaluate -> explain` | Components are reusable and versioned [S10] |
| Experiment tracking | MLflow (workspace is MLflow-compatible) | Params, metrics, artifacts | Use MLflow logging; v2 SDK has no logging API [S11] |
| Model registry | AML workspace registry; **AML registry** for dev -> prod sharing | Versioned MLflow-format models | `models.share()` is experimental [S10] |
| Online serving | **Managed online endpoint** | Real-time scoring, blue/green, mirroring | Entra (`aad_token`) auth for prod [S3] |
| Batch serving | **Batch endpoint** | Nightly scoring of the customer base | MLflow models need no scoring script [S5] |
| Scheduling | AML schedules (cron/recurrence); Azure Data Factory or Fabric optional | Retraining, batch scoring | AML v2 schedules have no event triggers [S6] |
| Prediction store | Azure SQL Database or Parquet in Blob | Scores + top reasons for campaigns/BI | Not an Azure ML feature; design choice |
| Container images | Azure Container Registry | Environments and any custom images | Endpoint identity needs pull rights [S4] |
| Alt. online serving | Azure Container Apps (FastAPI) | Own API code, scale to zero | Revisions + traffic split [S12] |
| Secrets | Azure Key Vault | Secrets and keys | |
| Identity | Microsoft Entra ID, managed identities, RBAC | Passwordless access | Separate *user identity* and *endpoint identity* [S4] |
| Networking | Workspace **managed virtual network**, private endpoints | Isolate inbound and outbound | Recommended over legacy per-deployment method [S13] |
| Service monitoring | Azure Monitor, Log Analytics, Application Insights | Latency, errors, logs, alerts | [S7] |
| Model monitoring | **Azure ML model monitoring** | Drift and quality signals | Needs data collection enabled [S2] |
| IaC | Bicep or Terraform | Reproducible environments | MLOps v2 samples use Terraform [S9] |
| Code quality | ruff, black, mypy, pytest, pre-commit | Quality gates | |

## 3. Environments
| Environment | Resource group | Purpose |
|---|---|---|
| `dev` | `rg-churn-dev` | Experiments, cheap compute |
| `staging` | `rg-churn-stg` | Pre-prod validation, smoke and load tests |
| `prod` | `rg-churn-prod` | Live service, private networking |

Use one Azure ML workspace per environment. Share the approved model from `dev` to an **Azure ML registry** so `staging` and `prod` consume the exact same model artifact [S10]. Registry assets are named and must exist in the registry before other registry assets (components, environments) can reference them [S10].

## 4. Production Steps

### Step 1: Subscription, RBAC, and cost guardrails
- Create resource groups, naming convention, and tags (`project`, `env`, `owner`).
- Entra groups with least-privilege roles; review the MLflow-operation permissions needed on the workspace [S11].
- Budgets and cost alerts per resource group.

### Step 2: Infrastructure as code
- Bicep (or Terraform) modules: storage, Key Vault, ACR, Log Analytics, Application Insights, Azure ML workspace, compute cluster, optional Azure SQL.
- **Prod workspace:** enable a **managed virtual network** (choose *allow only approved outbound* for production) so managed online endpoints and compute use it automatically; Azure ML creates private endpoints to the workspace's storage, Key Vault and ACR, and still allows Microsoft Container Registry access, which MLflow no-code deployment relies on [S13].
- If you use an Azure ML registry with a secured workspace, add private endpoints for the registry, its storage, and its ACR [S13].
- Deploy `dev` first, then reuse the same templates for `staging` and `prod`. The MLOps v2 accelerator has ready-made GitHub Actions workflows for infra deployment [S9].

### Step 3: Data ingestion, versioning, and validation
- Land the Orange/BigML churn CSV (OpenML 40701) in `raw/` (never edited) and register it as a versioned **Data Asset**.
- Run the pandera/Great Expectations contract check as the first pipeline component; fail the job on violations.
- For recurring production data, copy from source systems into Blob with Azure Data Factory (or a scheduled AML data import job, which is preview [S6]).

### Step 4: Training as an Azure ML pipeline
- Components: `validate -> split -> features -> train_catboost -> tune (Optuna) -> calibrate -> evaluate -> explain (SHAP)`, run as a pipeline job on the CPU cluster (min nodes 0).
- Log with **MLflow** (not the v1 SDK) so training stays portable [S11]. Install `mlflow` and `azureml-mlflow` (or `mlflow-skinny` for tracking only) [S11].
- Pin environments (conda/Docker) and fix seeds.

### Step 5: Package and register the model
- **(design choice)** Log the whole thing as one **MLflow pyfunc** model: preprocessing + CatBoost + calibrator + threshold + SHAP reasons. `predict()` returns a flat, tabular result (`churn_probability`, `decision`, `reason_1..3`). A custom pyfunc keeps *no-code deployment* available [S1] while still returning more than a raw prediction, and flat outputs suit model monitoring.
- The model's `conda.yaml` must include `azureml-inference-server-http` and `azureml-ai-monitoring` when conda dependencies are specified [S1].
- Register in the workspace with tags (data version, metrics, git SHA, threshold).
- **Promotion gate** in the pipeline: must beat the current production model on PR-AUC and pass calibration and segment-performance checks before it is shared to the registry.
- Share to the **Azure ML registry** (`models.share()` is experimental; or `az ml model create --registry-name`) [S10].

### Step 6: Deploy the online endpoint (managed online endpoint)
- Endpoint **`auth_mode: aad_token`** for production. It is the highest-security mode, supported on **managed** endpoints only, and requires the `score/action` permission (e.g., `AzureML Data Scientist`) for callers. Key auth is for development and testing [S3].
- Deployment **`blue`** from the registered MLflow model; enable **data collection** and **Application Insights** at deployment time. For MLflow models, data collection is a one-click option that creates `<endpoint>-<deployment>-model_inputs` and `...-model_outputs` data assets [S2].
- If you need a scoring script instead of no-code, you must also supply the environment (with `azureml-inference-server-http`, `azureml-ai-monitoring`) [S1].
- Note: local-endpoint testing is not supported for MLflow no-code deployments, so validate in `dev`/`staging` instead [S1].
- Network: set the endpoint's `public_network_access` to `disabled` so it is reachable only through the workspace private endpoint; outbound goes through the workspace managed VNet [S13].
- **Alternative: Azure Container Apps (FastAPI).** Use multiple-revision mode and traffic splitting (weights must total 100%); apps scale to zero unless they scale on CPU/memory [S12]. Choose this if you want to show your own API code; you then handle auth and monitoring yourself.

### Step 7: Safe rollout (blue/green)
1. Create `green` with 0% traffic and test it directly (`--deployment-name green`, or the `azureml-model-deployment` header) [S14].
2. **Mirror** (shadow) a share of live traffic to `green`: clients still get `blue` results. Limits: max 50% mirrored, one shadow deployment, not for Kubernetes endpoints, and requests that name a deployment are not mirrored [S14].
3. Shift live traffic gradually (e.g., `blue=90 green=10`); total must be 0 or 100 [S14].
4. Move to `green=100`, then delete `blue`. **Rollback** = set traffic back to `blue` [S14].
- Mind the default endpoint bandwidth quota (5 MBps) when load testing and mirroring [S14].

### Step 8: Batch scoring
- Create a **batch endpoint** for the same MLflow model. For MLflow models, Azure ML autogenerates the scoring pipeline, so no scoring script is needed [S5].
- Avoid a custom batch driver: it requires `azureml-core` (SDK v1), which reached end of support on 2026-06-30 [S5].
- Input: customer snapshot as a Data Asset/folder. Work is parallelized **by file**, so split large snapshots into several files [S5].
- Output: must go to a **blob-backed datastore** and the output path must be **unique per run**, or the job fails [S5]. A follow-up step loads results into Azure SQL / Parquet for campaigns and Power BI.
- Trigger nightly with an external orchestrator (Azure Data Factory or Fabric) by deploying the pipeline job under the batch endpoint [S6], or with a scheduled **Container Apps job** if you chose that stack [S12].

### Step 9: CI/CD with GitHub Actions
- **Auth:** `azure/login` with **OpenID Connect**: a Microsoft Entra app or **user-assigned managed identity** with a federated credential trusting your repo; store only client ID, tenant ID and subscription ID as secrets (no passwords) [S8]. Grant the identity least-privilege RBAC.
- **Pull request:** lint, type-check, unit and model-quality tests.
- **Merge to main:** run the training pipeline in `dev` with the AML CLI v2 (`az ml job create`), register the model, deploy to `staging`, run smoke and load tests [S9].
- **Release:** GitHub Environments with manual approval, then blue/green to `prod` (Step 7).
- Use the **Azure/mlops-v2** accelerator as the starting scaffold [S9].

### Step 10: Security and governance
- Entra token auth on endpoints, RBAC, managed identities (the *endpoint identity* pulls images from ACR and reads storage; the *user identity* calls the endpoint) [S4].
- Secrets only in Key Vault; no secrets in code or workflow files.
- Do not log raw customer identifiers; the data collector writes inputs/outputs to the workspace blob store, so apply the same access controls and retention rules to it **(design choice)**.
- Enable diagnostic settings so `AmlOnlineEndpointTrafficLogs` records the auth mode and caller identity [S3].
- Azure Policy to enforce network isolation settings [S15].

### Step 11: Monitoring and alerting
**Service health** [S7]
- Endpoint metrics (request latency, requests/min, status codes) and deployment metrics (CPU, memory) in Azure Monitor; alerts and dashboards on them.
- Application Insights (live metrics, failures, performance) and Log Analytics (Kusto) for logs.

**Model health** [S2]
- Prerequisite: data collection enabled on the deployment. The out-of-box monitor tracks **data drift, prediction drift, and data quality**, compares against recent production data, uses default thresholds, and runs on a schedule; create it with `az ml schedule create` (the monitoring YAML schema is in preview) [S2].
- Better for churn **(design choice)**: set the reference to **training data**, tune thresholds (e.g., PSI), and focus on top-N important features.
- Alerts go to email by default and can use **Event Grid**.
- **Model performance signal:** the data collector assigns a `correlationid` per row; log your own unique customer/prediction ID, then join true churn labels when they arrive (30-60 days later) to compute real PR-AUC and recall [S2].
- Batch endpoints and models outside Azure ML are not auto-collected: you must collect and register the production data yourself before monitoring it [S2].
- Data collector limits: online endpoints only; the SDK logs pandas DataFrames only [S2].

### Step 12: Retraining loop
- **Scheduled:** an AML schedule (recurrence or cron) on the training pipeline job, e.g., monthly. Cron `DAYS` and `MONTHS` fields are ignored, and v2 schedules have **no event-based triggers**; the schedule owner must keep run permission on the job [S6].
- **Alert-driven:** route monitor alerts via Event Grid to a workflow that starts the training pipeline (e.g., a GitHub Actions `workflow_dispatch` or Logic App) **(design choice)**.
- New model -> promotion gate -> `staging` -> mirror and canary in `prod` (Step 7) -> promote or discard.

### Step 13: Operations
- Runbook: rollback (traffic shift), failed batch rerun, drift alert triage, identity/credential rotation.
- Model card and data dictionary kept in the repo and linked from model tags.
- Monthly cost review; keep compute min nodes at 0.

## 5. Cost-Conscious Setup (portfolio / low budget)
| Choice | Why |
|---|---|
| `dev` + `prod` only (skip a full staging replica) | Fewer always-on resources |
| Compute cluster min nodes = 0 | Pay only while training |
| 1 instance managed endpoint, or Container Apps with scale to zero | Managed endpoints keep instances running; Container Apps cost nothing while scaled to zero [S12] |
| Batch endpoint on a small cluster, nightly | Cheaper than keeping online capacity for campaign lists |
| Skip managed VNet / private endpoints in the demo; document them as production hardening | Added cost and setup time |
| Tear down with `az group delete` after demos | Avoid idle spend |

## 6. Milestones
| Phase | Deliverable |
|---|---|
| 1 | Resource groups, IaC, RBAC, budgets |
| 2 | Data Assets, validation, AML training pipeline with MLflow tracking |
| 3 | Pyfunc model registered, promotion gate, registry sharing |
| 4 | Online endpoint (aad_token) + batch endpoint in staging |
| 5 | GitHub Actions CI/CD with OIDC, blue/green and mirroring to prod |
| 6 | Data collection, model monitoring, alerts, retraining schedule, runbook |

## 7. Risks and Mitigations
- **SDK v1 retirement:** AML SDK v1 was deprecated 2025-03-31 and reached end of support 2026-06-30 [S5]. Use only CLI/SDK v2 and MLflow; avoid custom batch drivers that need `azureml-core`.
- **Preview features:** model-monitoring YAML schema and data-import schedules are preview [S2][S6]; pin CLI versions and keep a manual drift report (Evidently) as a fallback **(design choice)**.
- **Cost overrun:** budgets, scale-to-zero compute, teardown scripts.
- **Training/serving skew:** serve the exact registered pyfunc pipeline, never re-implemented transforms.
- **Delayed labels hide decay:** drift signals as early warning; compute true performance when labels arrive via `correlationid` [S2].
- **Bandwidth throttling during mirroring/load tests:** default 5 MBps endpoint quota [S14].
- **Secret leakage:** Entra token auth, managed identities, OIDC, Key Vault.
- **Vendor lock-in:** MLflow model format and containers keep the model portable.

## 8. Verification Checklist
- [ ] IaC deploys cleanly to `dev`, then `prod`, without portal edits.
- [ ] Training pipeline runs end to end and logs to workspace MLflow.
- [ ] Registered pyfunc model gives identical outputs to the local pipeline on a fixture set.
- [ ] Online endpoint uses `aad_token`; unauthenticated and unauthorized calls return 401/403 [S3].
- [ ] Endpoint returns probability, decision, and top reasons; p95 latency < 100 ms under load test.
- [ ] Blue/green test: mirror 10%, split 90/10, full cutover, and rollback all exercised [S14].
- [ ] Batch job writes to a unique blob output path and results land in the prediction store [S5].
- [ ] Data collection creates `model_inputs` / `model_outputs` assets; monitor schedule runs and a forced threshold breach sends an alert [S2].
- [ ] CI/CD authenticates by OIDC with no stored passwords [S8].
- [ ] Retraining schedule triggers a pipeline run [S6].

## 9. Sources (Microsoft Learn, accessed 2026-10-06)
| ID | Topic | URL |
|---|---|---|
| S1 | Guidelines for deploying MLflow models; MLflow online deployment | https://learn.microsoft.com/azure/machine-learning/how-to-deploy-mlflow-models?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints?view=azureml-api-2 |
| S2 | Model monitoring, data collection | https://learn.microsoft.com/azure/machine-learning/concept-model-monitoring?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-monitor-model-performance?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/concept-data-collection?view=azureml-api-2 |
| S3 | Online endpoint authentication | https://learn.microsoft.com/azure/machine-learning/how-to-authenticate-online-endpoint?view=azureml-api-2 |
| S4 | Endpoint identity and permissions | https://learn.microsoft.com/azure/machine-learning/concept-endpoints-online-auth?view=azureml-api-2 |
| S5 | Batch endpoints; MLflow in batch | https://learn.microsoft.com/azure/machine-learning/how-to-use-batch-model-deployments?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-mlflow-batch?view=azureml-api-2 |
| S6 | Schedule pipeline jobs | https://learn.microsoft.com/azure/machine-learning/how-to-schedule-pipeline-job?view=azureml-api-2 |
| S7 | Monitor online endpoints | https://learn.microsoft.com/azure/machine-learning/how-to-monitor-online-endpoints?view=azureml-api-2 |
| S8 | Azure Login action with OIDC | https://learn.microsoft.com/azure/developer/github/connect-from-azure-openid-connect |
| S9 | MLOps v2 architecture, MLOps with GitHub, GitHub Actions with Azure ML | https://learn.microsoft.com/azure/architecture/ai-ml/guide/machine-learning-operations-v2 · https://learn.microsoft.com/azure/machine-learning/how-to-setup-mlops-github-azure-ml?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-github-actions-machine-learning?view=azureml-api-2 |
| S10 | Registries: share models/components | https://learn.microsoft.com/azure/machine-learning/how-to-share-models-pipelines-across-workspaces-with-registries?view=azureml-api-2 |
| S11 | MLflow and Azure Machine Learning; tracking | https://learn.microsoft.com/azure/machine-learning/concept-mlflow?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow-cli-runs?view=azureml-api-2 |
| S12 | Azure Container Apps overview, traffic splitting, scaling | https://learn.microsoft.com/azure/container-apps/overview · https://learn.microsoft.com/azure/container-apps/traffic-splitting · https://learn.microsoft.com/azure/container-apps/scale-app |
| S13 | Network isolation for managed online endpoints | https://learn.microsoft.com/azure/machine-learning/concept-secure-online-endpoint?view=azureml-api-2 · https://learn.microsoft.com/azure/machine-learning/how-to-secure-online-endpoint?view=azureml-api-2 |
| S14 | Safe rollout of online deployments | https://learn.microsoft.com/azure/machine-learning/how-to-safely-rollout-online-endpoints?view=azureml-api-2 |
| S15 | Network isolation planning, Azure Policy | https://learn.microsoft.com/azure/machine-learning/how-to-network-isolation-planning?view=azureml-api-2 |
