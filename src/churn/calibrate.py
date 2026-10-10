"""Calibration and business-cost threshold (Step 9).

    python -m churn.calibrate --config configs/params.yaml --data-dir data/processed \
        --tuning-dir reports/tuning --out-dir reports/calibration --model-dir models

Out-of-fold probabilities from the tuned CatBoost are used to pick a calibrator (none, sigmoid or
isotonic) and the decision threshold, so nothing is fitted on rows the model predicted from memory.
The final model is then fitted on the whole training split. The test set is never read.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from churn.calibrator import METHODS, Calibrator  # noqa: E402
from churn.config import BusinessParams, Params, Settings, load_params  # noqa: E402
from churn.features import derive_features  # noqa: E402
from churn.split import TRAIN_FILE, load_train, make_cv  # noqa: E402
from churn.train import CATBOOST_FIT_PARAMS, build_models, file_sha256, git_sha  # noqa: E402
from churn.tune import BEST_PARAMS_FILE  # noqa: E402

THRESHOLDS = np.round(np.arange(0.02, 0.95, 0.01), 2)
PLATEAU_TOLERANCE = 0.005  # within 0.5% of the best value counts as equally good


def tuned_pipeline(params: Params, tuning_dir: Path) -> Any:
    """Plain CatBoost pipeline with the parameters saved by `churn.tune`."""
    best = json.loads((tuning_dir / BEST_PARAMS_FILE).read_text(encoding="utf-8"))["best_params"]
    return clone(build_models(params)["catboost"]).set_params(
        **{f"model__{k}": v for k, v in best.items()}
    )


def oof_probabilities(model: Any, X: pd.DataFrame, y: pd.Series, params: Params) -> np.ndarray:
    """Out-of-fold churn probabilities, averaged over the CV repeats."""
    total, count = np.zeros(len(X)), np.zeros(len(X))
    for fit_idx, val_idx in make_cv(params).split(X, y):
        fold = clone(model).fit(X.iloc[fit_idx], y.iloc[fit_idx], **CATBOOST_FIT_PARAMS)
        total[val_idx] += fold.predict_proba(X.iloc[val_idx])[:, 1]
        count[val_idx] += 1
    return total / count


def cross_fitted(method: str, p: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Calibrated probabilities where each row is calibrated by a fit that did not see it."""
    out = np.zeros(len(p))
    for fit_idx, val_idx in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        out[val_idx] = Calibrator(method).fit(p[fit_idx], y[fit_idx]).predict(p[val_idx])
    return out


def compare_calibrators(
    p: np.ndarray, y: np.ndarray, seed: int
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Cross-validated Brier score and log loss per method, plus the cross-fitted probabilities."""
    scores, fitted = {}, {}
    for method in METHODS:
        q = np.clip(cross_fitted(method, p, y, seed), 1e-6, 1 - 1e-6)
        fitted[method] = q
        scores[method] = {"brier": float(brier_score_loss(y, q)), "log_loss": float(log_loss(y, q))}
    return scores, fitted


def retention_value(
    p: np.ndarray,
    y: np.ndarray,
    clv: np.ndarray,
    business: BusinessParams,
    threshold: float,
) -> float:
    """Realized value of making an offer to everyone with p >= threshold.

    A flagged churner is saved with probability `acceptance_rate` (worth their CLV); every flagged
    customer costs `offer_cost`.
    """
    flagged = p >= threshold
    saved = y[flagged] * business.acceptance_rate * clv[flagged]
    return float(saved.sum() - business.offer_cost * flagged.sum())


def choose_threshold(
    p: np.ndarray, y: np.ndarray, clv: np.ndarray, business: BusinessParams
) -> dict[str, float]:
    """Threshold that maximizes retention value, with the numbers behind it.

    When the value curve is flat, the exact maximum is noise, so among thresholds within
    `PLATEAU_TOLERANCE` of the best value the one closest to the break-even probability
    (offer cost / (acceptance x mean CLV), optimal for calibrated probabilities) is chosen.
    """
    values = np.array([retention_value(p, y, clv, business, t) for t in THRESHOLDS])
    break_even = float(business.offer_cost / (business.acceptance_rate * clv.mean()))
    near_best = THRESHOLDS[values >= values.max() - PLATEAU_TOLERANCE * abs(values.max())]
    best = float(near_best[np.argmin(np.abs(near_best - break_even))])
    value = retention_value(p, y, clv, business, best)
    flagged = p >= best
    return {
        "threshold": best,
        "value": value,
        "value_per_customer": value / len(p),
        "max_value_on_grid": float(values.max()),
        "value_flag_everyone": retention_value(p, y, clv, business, 0.0),
        "break_even_probability": break_even,
        "flagged_share": float(flagged.mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "recall": float(y[flagged].sum() / y.sum()),
    }


def threshold_plot(values: list[float], best: float, break_even: float, path: Path) -> None:
    """Retention value against the decision threshold."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(THRESHOLDS, values)
    ax.axvline(best, color="green", linestyle="--", label=f"chosen {best:.2f}")
    ax.axvline(break_even, color="gray", linestyle=":", label=f"break-even {break_even:.2f}")
    ax.set(xlabel="Threshold", ylabel="Total retention value", title="Value by threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def reliability_plot(
    y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, method: str, path: Path
) -> None:
    """Reliability curves of the raw and the chosen calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, q in (("raw", raw), (f"calibrated ({method})", calibrated)):
        frac, mean = calibration_curve(y, q, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set(xlabel="Predicted probability", ylabel="Observed churn rate", title="Reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(
    params: Params,
    config_path: Path,
    data_dir: Path,
    tuning_dir: Path,
    out_dir: Path,
    model_dir: Path,
    settings: Settings,
) -> dict[str, Any]:
    """Pick the calibrator and threshold from OOF predictions, fit the final model, save all."""
    train = load_train(data_dir)
    X, y = train.drop(columns=params.data.target), train[params.data.target]
    labels = y.to_numpy()
    clv = derive_features(X)["total_charge"].to_numpy() * params.business.remaining_months

    model = tuned_pipeline(params, tuning_dir)
    raw = oof_probabilities(model, X, y, params)
    scores, fitted = compare_calibrators(raw, labels, params.seed)
    method = min(scores, key=lambda m: scores[m]["brier"])
    threshold = choose_threshold(fitted[method], labels, clv, params.business)

    final = clone(model).fit(X, y, **CATBOOST_FIT_PARAMS)
    calibrator = Calibrator(method).fit(raw, labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_dir / "final_model.joblib")
    joblib.dump(calibrator, model_dir / "calibrator.joblib")
    reliability_plot(labels, raw, fitted[method], method, out_dir / "reliability.png")
    values = [retention_value(fitted[method], labels, clv, params.business, t) for t in THRESHOLDS]
    threshold_plot(
        values,
        threshold["threshold"],
        threshold["break_even_probability"],
        out_dir / "threshold_value.png",
    )
    result: dict[str, Any] = {
        "method": method,
        "calibrators": {m: {k: round(v, 5) for k, v in s.items()} for m, s in scores.items()},
        "decision": {k: round(v, 4) for k, v in threshold.items()},
        "assumptions": params.business.model_dump(),
        "value_by_threshold": {
            f"{t:.2f}": round(v)
            for t, v in zip(THRESHOLDS, values, strict=True)
            if round(t * 100) % 5 == 0
        },
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="calibration"):
        mlflow.set_tags(
            {
                "git_sha": git_sha(),
                "data_sha256": file_sha256(data_dir / TRAIN_FILE),
                "stage": "calibration",
                "calibration_method": method,
            }
        )
        mlflow.log_metrics({f"{m}_brier": s["brier"] for m, s in scores.items()})
        mlflow.log_metrics({f"decision_{k}": v for k, v in threshold.items()})
        mlflow.log_params(params.business.model_dump())
        mlflow.log_artifacts(str(out_dir))
        mlflow.log_artifact(str(config_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("config", "data-dir", "tuning-dir", "out-dir", "model-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    result = run(
        load_params(a.config),
        a.config,
        a.data_dir,
        a.tuning_dir,
        a.out_dir,
        a.model_dir,
        Settings(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
