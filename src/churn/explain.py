"""Explainability (Step 11): global SHAP plots and per-customer reasons.

    python -m churn.explain --config configs/params.yaml --data-dir data/processed \
        --model-dir models --out-dir reports/explainability

SHAP values come from CatBoost's exact TreeSHAP (`ShapValues`), so they handle the categorical
columns natively and add up to the model's raw score. Global plots use the training split; the
test split is never read. `top_reasons` is the per-customer logic the served model reuses.
"""

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from catboost import Pool  # noqa: E402

from churn.config import Params, Settings, load_params  # noqa: E402
from churn.features import catboost_cat_features  # noqa: E402
from churn.split import TRAIN_FILE, load_train  # noqa: E402
from churn.train import file_sha256, git_sha  # noqa: E402

N_REASONS = 3
# Smaller pushes (about +10% on the odds of churn) are noise, not reasons to act on.
MIN_CONTRIBUTION = 0.1
N_EXAMPLES = 10


def shap_contributions(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Engineered features and their SHAP contributions to the log-odds of churn.

    `model` is the fitted pipeline (feature step, then CatBoost). Row i's contributions plus the
    model's base value equal its raw score.
    """
    features = model[:-1].transform(X)
    pool = Pool(features, cat_features=catboost_cat_features())
    values = model[-1].get_feature_importance(pool, type="ShapValues")
    return features, values[:, :-1]


def _describe(name: str, value: object) -> str:
    return f"{name}={value:.1f}" if isinstance(value, float) else f"{name}={value}"


def top_reasons(model: Any, X: pd.DataFrame, k: int = N_REASONS) -> pd.DataFrame:
    """The `k` features pushing each customer's churn risk up the most, as `reason_1..k`.

    Only features whose contribution is at least `MIN_CONTRIBUTION` are listed, so customers with
    no clear risk driver get empty slots.
    """
    features, contributions = shap_contributions(model, X)
    rows = []
    for values, contrib in zip(features.to_numpy(dtype=object), contributions, strict=True):
        best = np.argsort(-contrib)[:k]
        rows.append(
            [
                _describe(features.columns[i], values[i]) if contrib[i] >= MIN_CONTRIBUTION else ""
                for i in best
            ]
        )
    return pd.DataFrame(rows, columns=[f"reason_{i + 1}" for i in range(k)], index=X.index)


def importance_plot(importance: pd.Series, path: Path) -> None:
    """Mean absolute SHAP value per feature (top 15)."""
    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.to_numpy())
    ax.set(xlabel="Mean |SHAP| (log-odds of churn)", title="Global feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def summary_plot(features: pd.DataFrame, contributions: np.ndarray, path: Path) -> None:
    """SHAP beeswarm for the numeric features (the categorical ones have no colour scale)."""
    numeric = [c for c in features.columns if c not in catboost_cat_features()]
    keep = [features.columns.get_loc(c) for c in numeric]
    shap.summary_plot(
        contributions[:, keep], features[numeric].astype(float), max_display=15, show=False
    )
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def run(
    params: Params,
    config_path: Path,
    data_dir: Path,
    model_dir: Path,
    out_dir: Path,
    settings: Settings,
) -> pd.DataFrame:
    """Write the global importance, plots and example reasons; return the importance table."""
    train = load_train(data_dir)
    X = train.drop(columns=params.data.target)
    model = joblib.load(model_dir / "final_model.joblib")

    features, contributions = shap_contributions(model, X)
    importance = pd.Series(np.abs(contributions).mean(axis=0), index=features.columns)
    importance = importance.sort_values(ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    importance.rename("mean_abs_shap").round(4).to_csv(out_dir / "global_importance.csv")
    importance_plot(importance, out_dir / "shap_importance.png")
    summary_plot(features, contributions, out_dir / "shap_summary.png")

    sample = X.sample(N_EXAMPLES, random_state=params.seed)
    examples = top_reasons(model, sample)
    examples.insert(0, "churn_probability", model.predict_proba(sample)[:, 1].round(3))
    examples.insert(1, "churned", train.loc[sample.index, params.data.target])
    examples.to_csv(out_dir / "example_reasons.csv")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="explainability"):
        mlflow.set_tags(
            {
                "git_sha": git_sha(),
                "data_sha256": file_sha256(data_dir / TRAIN_FILE),
                "stage": "explainability",
            }
        )
        mlflow.log_artifacts(str(out_dir))
        mlflow.log_artifact(str(config_path))
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("config", "data-dir", "model-dir", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    importance = run(
        load_params(a.config), a.config, a.data_dir, a.model_dir, a.out_dir, Settings()
    )
    print(importance.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
