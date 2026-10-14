# Using the churn model

The final model is one MLflow pyfunc model: feature pipeline, CatBoost, calibrator, decision threshold and the reasons logic. `make package` builds and registers it (`src/churn/package.py`; the served code is `src/churn/pyfunc_model.py`); the REST API (branch 12) serves the same model.

## Register

```bash
make package
```

It refuses to register unless the test-set gate passes (`reports/evaluation/test_evaluation.json`, criteria in `params.yaml`), logs the model with a pinned `conda.yaml`/`requirements.txt` and the `churn` code, registers it as `churn-catboost` (name from `MLFLOW_MODEL_NAME`), sets the alias `production`, then reloads the registered model and checks that it reproduces the local pipeline's output on 200 customers.

## Predict from Python

```python
import mlflow, pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")          # MLFLOW_TRACKING_URI in .env
model = mlflow.pyfunc.load_model("models:/churn-catboost@production")
print(model.predict(customers))                          # customers: a DataFrame, see below
```

**Input:** one row per customer with the 19 predictor columns of `docs/data_contract.md`: `state` and `area_code` as strings, the other columns as integers or floats exactly as in the data contract. No `phone_number` and no `class`. A missing column raises an error.

**Output:** one row per customer, in the same order.

| Column | Meaning |
|---|---|
| `churn_probability` | Calibrated probability of churn (0-1) |
| `decision` | 1 if `churn_probability` is at or above the threshold (0.23), else 0 |
| `reason_1..3` | Up to three features raising this customer's risk by at least 0.1 log-odds, as `feature=value`; empty when there is no clear driver |

## Notes

- Reasons name what raises risk, not whether the customer will churn. Read them with the probability and decision, and act on them for flagged customers.
- The threshold comes from the business assumptions in `params.yaml` (`business`); change them, then rerun `make calibrate` and `make package`.
- The registry is a local SQLite file (`mlflow.db`) with artifacts under `mlruns/`, both git-ignored. A model must be re-registered on a new machine (`make calibrate` then `make package`).
- Measured in-process on this machine: a single-row prediction takes about 20 ms (p95 23 ms) and a 1,000-row batch about 125 ms. The 100 ms target applies to the HTTP service and is measured by the load test in branch 12.

## REST API

`api/main.py` serves the registered model over HTTP.

```bash
make serve              # uvicorn on http://127.0.0.1:8000 (loads models:/churn-catboost@production)
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | One customer (the 19 fields of the data contract) in, `{churn_probability, decision, reasons}` out |
| `POST /predict/batch` | `{"customers": [...]}` (1 to 1,000) in, `{"predictions": [...]}` out, in the same order |
| `GET /health` | `{"status": "ok"}` once the model is loaded (503 otherwise) |
| `GET /model-info` | Model URI, MLflow run id, decision threshold, input columns |

Requests are validated against the data contract (`api/schemas.py`): service calls above 9, an unknown area code, a state outside 0-50, a negative amount, a missing field, or any extra field (`phone_number`, `class`) give HTTP 422. `state` and `area_code` may be sent as strings or numbers. Interactive docs are at `/docs`.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @customer.json
```

## Docker

```bash
make package            # registers the model and exports it to build/model
make docker-build       # builds churn-api:local (about 330 MB of content, runs as a non-root user)
make docker-run         # serves on http://127.0.0.1:8000
```

The image contains only the API code and the exported model folder (which bundles the `churn` code and pins its packages); it installs `mlflow-skinny` and none of the training libraries. Settings are environment variables: `MODEL_URI` (default `/app/model` in the image) and `WEB_CONCURRENCY` (worker processes, default 4).

## Load test

`make loadtest` (with the service running) drives `/predict` and `/predict/batch` with 10 simulated users for 60 seconds. Results and the latency-vs-load curve are in `reports/load_test.md`.
