# Load test of the churn API

Tool: Locust (`tests/load/locustfile.py`, `make loadtest`). Target (spec section 2): inference p95 latency below 100 ms.

## Setup

- **Service:** the Docker image `churn-api:local` (`Dockerfile`), serving the registered model through uvicorn with 2 or 4 worker processes.
- **Traffic:** each simulated user sends `POST /predict` (90% of requests) or `POST /predict/batch` with 10 customers (10%), then waits 50-200 ms. Customers are random but valid under `docs/data_contract.md`; no real data is used.
- **Duration:** 45 seconds per run, users ramped up at once. Zero requests failed in every run below.
- **Machine:** one Windows laptop (8 logical CPUs) running Docker Desktop, Locust and other applications at the same time. Locust and the container compete for the same CPUs and the Docker port forwarding adds overhead, so these figures are pessimistic compared with a dedicated server. Each row is a single run; the tail (p95 and p99) varies from run to run by tens of milliseconds.

## Results

Latency of all requests, in milliseconds (`/predict` and `/predict/batch` pooled):

| Workers | Users | Requests/s | Median | p95 | p99 |
|---|---|---|---|---|---|
| 2 | 1 | 6.2 | 22 | 44 | 180 |
| 2 | 3 | 15.9 | 36 | 140 | 340 |
| 2 | 5 | 25.5 | 45 | 130 | 240 |
| 2 | 10 | 45.0 | 73 | 190 | 270 |
| **4 (default)** | 1 | 5.8 | 29 | 77 | 230 |
| **4 (default)** | 3 | 18.1 | 27 | **69** | 110 |
| **4 (default)** | 5 | 28.9 | 33 | **100** | 150 |
| **4 (default)** | 10 | 50.1 | 54 | 160 | 220 |

For comparison, a single uvicorn process on the host (no container) with 10 users gave p95 of 250 ms (first version of the scoring code) and 230 ms (features computed once per request).

## Read-out

- **The p95 target is met at light load and not at heavy load.** With the default 4 workers, p95 is 69-77 ms for 1-3 users (up to about 18 requests/s), 100 ms at 5 users (about 29 requests/s, borderline) and 160 ms at 10 users (about 50 requests/s). The 10-user run is a demanding test: each user sends a request every 50-200 ms.
- **More workers help only up to a point.** 2 workers gave 130-140 ms at 3-5 users; 4 workers gave 69-100 ms. 4 workers did not improve the 10-user result, and cost about 730 MB of memory against about 420 MB.
- **Where one request spends its time** (single row, measured in-process before the features were computed once): feature engineering 4.4 ms (it ran twice per request), CatBoost prediction about 2.4 ms, SHAP reasons about 8 ms, MLflow wrapper about 1 ms, 20 ms in total. The SHAP reasons are the largest single cost.
- **The tail is noisy** (p99 of 110-340 ms, even with 1 user in some runs), which points at the shared test machine more than at the service.

## Not done / options

- Reasons are only useful for flagged customers (see `reports/explainability.md`). Skipping the SHAP step for customers below the threshold would save about 8 ms on roughly 88% of requests, at the cost of empty reasons for unflagged customers. This changes what the API returns, so it was not done.
- Feature engineering is 4.4 ms for a single row and could be sped up.
- A load test on a dedicated machine, with Locust on a different host, would give a cleaner figure.
