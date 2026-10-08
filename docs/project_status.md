# Project Status

Update this file in the same commit as each code change.

## v1 (local build)

| # | Branch | Steps | Status |
|---|---|---|---|
| 1 | `feature/project-scaffold` | setup | Merged |
| 2 | `feature/data-ingestion-validation` | 1-2 | Merged |
| 3 | `feature/train-test-split` | 4 | Merged |
| 4 | `feature/eda` | 3 | Merged |
| 5 | `feature/feature-pipeline` | 5 | Merged |
| 6 | `feature/baselines-imbalance` | 6-7 | PR open (into `feature/model-training`): `make train`, 5 variants in MLflow |
| 7 | `feature/tuning` | 8 | Not started |
| 8 | `feature/calibration-threshold` | 9 | Not started |
| 9 | `feature/final-evaluation` | 10 | Not started |
| 10 | `feature/explainability` | 11 | Not started |
| 11 | `feature/pyfunc-model` | 12 | Not started |
| 12 | `feature/api-docker-loadtest` | 12 | Not started |

## Environment notes

- CI: `.github/workflows/ci.yml` runs `make check` (ruff, black, mypy, pytest) on every PR and on pushes to `main`. Model training and quality gates stay local.
- Python 3.11 in `.venv`. `shap` is pinned to 0.51.0 because 0.52.0 requires Python 3.12+.
- On Windows, `make` (winget `ezwinports.make`) may need a new terminal to appear on `PATH`.
- Parquet is unavailable here (Windows Application Control blocks pyarrow's Parquet DLL), so interim data is CSV; read it with `churn.data.load_interim`.
- `orjson` (left behind by the removed DVC) is blocked by Windows Application Control and breaks `import mlflow` through FastAPI; it was uninstalled. Do not reinstall it.
