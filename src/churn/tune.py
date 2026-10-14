"""Optuna tuning of plain CatBoost (Step 8), logged to MLflow.

    python -m churn.tune --config configs/params.yaml --data-dir data/processed \
        --out-dir reports/tuning

Trials are scored by PR-AUC with a cheaper CV than the final one; the best parameters are then
re-scored with the full repeated CV so the result compares directly with the baseline.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from churn.config import Params, Settings, load_params
from churn.runinfo import file_sha256, git_sha
from churn.split import TRAIN_FILE, load_train
from churn.train import CATBOOST_FIT_PARAMS, build_models, evaluate_model

BEST_PARAMS_FILE = "best_params.json"


def suggest_params(trial: optuna.Trial) -> dict[str, float | int]:
    """CatBoost search space, leaning to stronger regularization (baseline train-CV gap is 0.10).

    `iterations` is tuned directly instead of using early stopping: stopping on the scored fold
    would make the CV score optimistic.
    """
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
    }


def make_objective(
    X: pd.DataFrame, y: pd.Series, params: Params
) -> Callable[[optuna.Trial], float]:
    """Objective: mean PR-AUC over the tuning CV folds, reporting each fold so bad trials prune."""
    base = build_models(params)["catboost"]
    cv = RepeatedStratifiedKFold(
        n_splits=params.tune.cv_folds, n_repeats=params.tune.cv_repeats, random_state=params.seed
    )

    def objective(trial: optuna.Trial) -> float:
        chosen = suggest_params(trial)
        scores: list[float] = []
        for step, (fit_idx, val_idx) in enumerate(cv.split(X, y)):
            model = clone(base).set_params(**{f"model__{k}": v for k, v in chosen.items()})
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx], **CATBOOST_FIT_PARAMS)
            proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(float(average_precision_score(y.iloc[val_idx], proba)))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def tune(X: pd.DataFrame, y: pd.Series, params: Params) -> optuna.Study:
    """Run the study; each trial is logged as a nested MLflow run."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=params.seed, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    objective = make_objective(X, y, params)

    def logged(trial: optuna.Trial) -> float:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            start = time.time()
            try:
                value = objective(trial)
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                mlflow.log_params(trial.params)
                mlflow.log_metric("seconds", time.time() - start)
                raise
            mlflow.set_tag("state", "complete")
            mlflow.log_params(trial.params)
            mlflow.log_metrics({"tune_pr_auc": value, "seconds": time.time() - start})
            return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(logged, n_trials=params.tune.n_trials)
    return study


def run(
    params: Params, config_path: Path, data_dir: Path, out_dir: Path, settings: Settings
) -> dict[str, object]:
    """Tune, re-score the best parameters with the full CV, write them to `out_dir`."""
    train = load_train(data_dir)
    X, y = train.drop(columns=params.data.target), train[params.data.target]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    start = time.time()
    with mlflow.start_run(run_name="tuning"):
        mlflow.set_tags(
            {
                "git_sha": git_sha(),
                "data_sha256": file_sha256(data_dir / TRAIN_FILE),
                "stage": "tuning",
            }
        )
        mlflow.log_params({"seed": params.seed, "n_trials": params.tune.n_trials})
        study = tune(X, y, params)

        best = study.best_params
        final = clone(build_models(params)["catboost"]).set_params(
            **{f"model__{k}": v for k, v in best.items()}
        )
        metrics = evaluate_model(final, X, y, params, "catboost")
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metrics(
            {
                **metrics,
                "tune_best_pr_auc": study.best_value,
                "trials_pruned": n_pruned,
                "total_seconds": time.time() - start,
            }
        )
        mlflow.log_artifact(str(config_path))

    result: dict[str, object] = {
        "best_params": best,
        "tune_best_pr_auc": round(study.best_value, 4),
        "final_cv": {k: round(v, 4) for k, v in metrics.items()},
        "trials": len(study.trials),
        "pruned": n_pruned,
        "total_seconds": round(time.time() - start),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / BEST_PARAMS_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(load_params(args.config), args.config, args.data_dir, args.out_dir, Settings())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
