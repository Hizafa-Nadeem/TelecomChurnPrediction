# Using the churn model

The final model is one MLflow pyfunc model: feature pipeline, CatBoost, calibrator, decision threshold and the reasons logic. `make package` builds and registers it (`src/churn/pyfunc_model.py`); the REST API (branch 12) serves the same model.

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
