"""Baselines and class-imbalance comparison (Steps 6-7), logged to MLflow.

    python -m churn.train --config configs/params.yaml --data-dir data/processed

Every variant is scored with the same repeated stratified CV on the training split only.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from churn.config import Params, Settings, load_params
from churn.features import FEATURE_COLUMNS, FLAG_COLUMNS, build_pipeline, catboost_cat_features
from churn.runinfo import file_sha256, git_sha
from churn.split import TRAIN_FILE, load_train, make_cv
from churn.validation import CATEGORICAL_COLUMNS

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
# CatBoost copies `cat_features` in its constructor, which breaks sklearn's clone() inside CV, so
# the categorical columns are passed at fit time instead.
CATBOOST_FIT_PARAMS = {"model__cat_features": catboost_cat_features()}


def build_models(params: Params) -> dict[str, BaseEstimator]:
    """The variants compared in Steps 6-7, each a full pipeline from the cleaned frame."""
    seed = params.seed

    def catboost(**kwargs: object) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=params.train.catboost_iterations,
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
            **kwargs,
        )

    smote_columns = [FEATURE_COLUMNS.index(c) for c in [*CATEGORICAL_COLUMNS, *FLAG_COLUMNS]]
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": Pipeline(
            [
                *build_pipeline("logreg").steps,
                (
                    "model",
                    LogisticRegressionCV(
                        Cs=params.train.logreg_c_grid,
                        scoring="average_precision",
                        class_weight="balanced",
                        l1_ratios=(0.0,),
                        use_legacy_attributes=False,
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline([*build_pipeline("catboost").steps, ("model", catboost())]),
        "catboost_weighted": Pipeline(
            [*build_pipeline("catboost").steps, ("model", catboost(auto_class_weights="Balanced"))]
        ),
        "catboost_smotenc": ImbPipeline(
            [
                *build_pipeline("catboost").steps,
                ("smote", SMOTENC(categorical_features=smote_columns, random_state=seed)),
                ("model", catboost()),
            ]
        ),
    }


def evaluate_model(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, params: Params, name: str
) -> dict[str, float]:
    """Repeated-CV metrics: mean and std per metric, plus the train-vs-validation gap."""
    fit_params = CATBOOST_FIT_PARAMS if name.startswith("catboost") else None
    scores = cross_validate(
        model, X, y, cv=make_cv(params), scoring=SCORING, return_train_score=True, params=fit_params
    )
    metrics: dict[str, float] = {}
    for metric in SCORING:
        val, train = scores[f"test_{metric}"], scores[f"train_{metric}"]
        metrics[f"cv_{metric}_mean"] = float(np.mean(val))
        metrics[f"cv_{metric}_std"] = float(np.std(val))
        metrics[f"train_{metric}_mean"] = float(np.mean(train))
        metrics[f"gap_{metric}"] = float(np.mean(train) - np.mean(val))
    return metrics


def run(params: Params, config_path: Path, data_dir: Path, settings: Settings) -> pd.DataFrame:
    """Score every variant, log one MLflow run each, and return the results table."""
    train = load_train(data_dir)
    X, y = train.drop(columns=params.data.target), train[params.data.target]
    data_hash = file_sha256(data_dir / TRAIN_FILE)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    rows = {}
    for name, model in build_models(params).items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"git_sha": git_sha(), "data_sha256": data_hash, "stage": "baselines"})
            mlflow.log_params(
                {
                    "model": name,
                    "seed": params.seed,
                    "cv_folds": params.split.cv_folds,
                    "cv_repeats": params.split.cv_repeats,
                }
            )
            rows[name] = evaluate_model(model, X, y, params, name)
            mlflow.log_metrics(rows[name])
            mlflow.log_artifact(str(config_path))
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    table = run(load_params(args.config), args.config, args.data_dir, Settings())
    print(table[["cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "gap_pr_auc"]].round(4))


if __name__ == "__main__":
    main()
