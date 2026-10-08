# Project Status

Update this file in the same commit as each code change.

## v1 (local build)

| # | Branch | Steps | Status |
|---|---|---|---|
| 1 | `feature/project-scaffold` | setup | In progress: scaffold, config models, `make check` |
| 2 | `feature/data-ingestion-validation` | 1-2 | Not started |
| 3 | `feature/train-test-split` | 4 | Not started |
| 4 | `feature/eda` | 3 | Not started |
| 5 | `feature/feature-pipeline` | 5 | Not started |
| 6 | `feature/baselines-imbalance` | 6-7 | Not started |
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
