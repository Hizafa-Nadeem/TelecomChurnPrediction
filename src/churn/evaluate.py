"""One-time test-set evaluation (Step 10).

    python -m churn.evaluate --config configs/params.yaml --data-dir data/processed \
        --model-dir models --calibration-dir reports/calibration --out-dir reports/evaluation

This is the only module allowed to read the test split. It scores the final model, calibrator and
threshold produced by `churn.calibrate` (nothing is re-fitted or re-picked here), adds bootstrap
confidence intervals, compares with the baselines and breaks the errors down by segment. A second
run is refused unless `--allow-rerun` is given, so the test set cannot be reused by accident.
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
from fairlearn.metrics import MetricFrame, selection_rate  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.calibrate import Calibrator, retention_value  # noqa: E402
from churn.config import Params, Settings, load_params  # noqa: E402
from churn.features import derive_features  # noqa: E402
from churn.split import TRAIN_FILE, load_test, load_train  # noqa: E402
from churn.train import build_models, file_sha256, git_sha  # noqa: E402

RESULT_FILE = "test_evaluation.json"
SERVICE_BINS = ([-1, 1, 3, 99], ["0-1", "2-3", "4+"])
N_BOOTSTRAP = 1000
PRECISION_FLOOR = 0.90


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion (sound for small groups)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def compute_metrics(
    y: np.ndarray,
    raw: np.ndarray,
    cal: np.ndarray,
    flagged: np.ndarray,
    clv: np.ndarray,
    params: Params,
    threshold: float,
) -> dict[str, float]:
    """All reported test metrics for one (re)sample of customers."""
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pr, rc, _ = precision_recall_curve(y, raw)
    top = np.argsort(-raw)[: max(1, len(y) // 10)]
    cal = np.clip(cal, 1e-6, 1 - 1e-6)
    return {
        "pr_auc": float(average_precision_score(y, raw)),
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc_calibrated": float(average_precision_score(y, cal)),
        "roc_auc_calibrated": float(roc_auc_score(y, cal)),
        "recall_at_precision_90": float(rc[pr >= PRECISION_FLOOR].max(initial=0.0)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flagged_share": float(flagged.mean()),
        "brier": float(brier_score_loss(y, cal)),
        "log_loss": float(log_loss(y, cal)),
        "lift_top_decile": float(y[top].mean() / y.mean()),
        "retention_value_per_customer": retention_value(cal, y, clv, params.business, threshold)
        / len(y),
    }


def bootstrap_intervals(
    y: np.ndarray,
    raw: np.ndarray,
    cal: np.ndarray,
    clv: np.ndarray,
    params: Params,
    threshold: float,
    baseline_raw: np.ndarray,
) -> tuple[dict[str, list[float]], list[float]]:
    """95% percentile intervals over resampled customers, and the paired PR-AUC gain vs baseline."""
    rng = np.random.default_rng(params.seed)
    flagged = cal >= threshold
    runs: list[dict[str, float]] = []
    gain: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        runs.append(
            compute_metrics(y[idx], raw[idx], cal[idx], flagged[idx], clv[idx], params, threshold)
        )
        gain.append(
            average_precision_score(y[idx], raw[idx])
            - average_precision_score(y[idx], baseline_raw[idx])
        )
    table = pd.DataFrame(runs)
    intervals = {
        c: [float(table[c].quantile(0.025)), float(table[c].quantile(0.975))] for c in table
    }
    return intervals, [float(np.quantile(gain, 0.025)), float(np.quantile(gain, 0.975))]


def state_groups(train: pd.DataFrame, target: str, k: int = 10) -> dict[str, str]:
    """High / low churn state groups, from the training split only."""
    rate = train.groupby("state", observed=True)[target].mean().sort_values()
    groups = {s: "middle" for s in rate.index}
    groups.update({s: "low-churn states" for s in rate.index[:k]})
    groups.update({s: "high-churn states" for s in rate.index[-k:]})
    return groups


def segment_table(
    X: pd.DataFrame, y: np.ndarray, flagged: np.ndarray, groups: dict[str, str]
) -> list[dict[str, Any]]:
    """Counts, recall and precision of the decision per segment (Fairlearn MetricFrame)."""
    segments: dict[str, pd.Series] = {
        "international_plan": X["international_plan"].map({0: "no", 1: "yes"}),
        "service_calls": pd.cut(
            X["number_customer_service_calls"], SERVICE_BINS[0], labels=SERVICE_BINS[1]
        ),
        "voice_mail_plan": X["voice_mail_plan"].map({0: "no", 1: "yes"}),
        "area_code": X["area_code"].astype(str),
        "state_group": X["state"].astype(str).map(groups),
    }
    rows = []
    for name, feature in segments.items():
        frame = MetricFrame(
            metrics={
                "recall": lambda t, p: recall_score(t, p, zero_division=0),
                "precision": lambda t, p: precision_score(t, p, zero_division=0),
                "flagged_share": selection_rate,
            },
            y_true=y,
            y_pred=flagged.astype(int),
            sensitive_features=feature.astype(str).to_numpy(),
        )
        for level, metrics in frame.by_group.iterrows():
            mask = feature.astype(str).to_numpy() == level
            churners, caught = int(y[mask].sum()), int((flagged[mask] & (y[mask] == 1)).sum())
            low, high = wilson_interval(caught, churners)
            rows.append(
                {
                    "segment": name,
                    "level": str(level),
                    "n": int(mask.sum()),
                    "churners": churners,
                    "churn_rate": round(churners / mask.sum(), 4),
                    "recall": round(float(metrics["recall"]), 4),
                    "recall_ci95": [round(low, 4), round(high, 4)],
                    "precision": round(float(metrics["precision"]), 4),
                    "flagged_share": round(float(metrics["flagged_share"]), 4),
                }
            )
    return rows


def evaluation_plot(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, path: Path) -> None:
    """Cumulative gain, lift by decile and the reliability of the calibrated probabilities."""
    order = np.argsort(-raw)
    caught = np.cumsum(y[order]) / y.sum()
    share = np.arange(1, len(y) + 1) / len(y)
    deciles = np.array_split(y[order], 10)
    lift = [d.mean() / y.mean() for d in deciles]
    bins = np.clip((cal * 10).astype(int), 0, 9)
    observed = [
        (y[bins == b].mean(), cal[bins == b].mean()) for b in range(10) if (bins == b).any()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(share, caught, label="model")
    axes[0].plot([0, 1], [0, 1], "k--", label="random")
    axes[0].set(xlabel="Share of customers contacted", ylabel="Share of churners caught")
    axes[0].set_title("Cumulative gain")
    axes[0].legend()
    axes[1].bar(range(1, 11), lift)
    axes[1].axhline(1, color="k", linestyle="--")
    axes[1].set(xlabel="Decile (1 = highest risk)", ylabel="Lift", title="Lift by decile")
    axes[2].plot([o[1] for o in observed], [o[0] for o in observed], marker="o")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set(xlabel="Predicted probability", ylabel="Observed churn rate", title="Reliability")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(
    params: Params,
    config_path: Path,
    data_dir: Path,
    model_dir: Path,
    calibration_dir: Path,
    out_dir: Path,
    settings: Settings,
    allow_rerun: bool = False,
) -> dict[str, Any]:
    """Evaluate once on the test split and write the results to `out_dir`."""
    result_path = out_dir / RESULT_FILE
    if result_path.exists() and not allow_rerun:
        raise FileExistsError(
            f"{result_path} exists: the test set was already evaluated. Use --allow-rerun only to "
            "repeat the identical evaluation after a code bug, never after changing the model."
        )
    target = params.data.target
    decision = json.loads((calibration_dir / "calibration.json").read_text(encoding="utf-8"))
    threshold = float(decision["decision"]["threshold"])
    model = joblib.load(model_dir / "final_model.joblib")
    calibrator: Calibrator = joblib.load(model_dir / "calibrator.joblib")

    train = load_train(data_dir)
    test = load_test(data_dir)
    X, y = test.drop(columns=target), test[target].to_numpy()
    raw = model.predict_proba(X)[:, 1]
    cal = calibrator.predict(raw)
    flagged = cal >= threshold
    clv = derive_features(X)["total_charge"].to_numpy() * params.business.remaining_months

    metrics = compute_metrics(y, raw, cal, flagged, clv, params, threshold)
    logreg = clone(build_models(params)["logreg"]).fit(train.drop(columns=target), train[target])
    logreg_raw = logreg.predict_proba(X)[:, 1]
    intervals, gain_ci = bootstrap_intervals(y, raw, cal, clv, params, threshold, logreg_raw)

    result: dict[str, Any] = {
        "threshold": threshold,
        "n_test": len(y),
        "churners": int(y.sum()),
        "confusion": {
            "tp": int((flagged & (y == 1)).sum()),
            "fp": int((flagged & (y == 0)).sum()),
            "fn": int((~flagged & (y == 1)).sum()),
            "tn": int((~flagged & (y == 0)).sum()),
        },
        "metrics": {k: {"value": v, "ci95": intervals[k]} for k, v in metrics.items()},
        "baselines": {
            "dummy_pr_auc": float(y.mean()),
            "logreg_pr_auc": float(average_precision_score(y, logreg_raw)),
            "logreg_roc_auc": float(roc_auc_score(y, logreg_raw)),
            "pr_auc_gain_vs_logreg": {
                "value": metrics["pr_auc"] - float(average_precision_score(y, logreg_raw)),
                "ci95": gain_ci,
            },
        },
        "segments": segment_table(X, y, flagged, state_groups(train, target)),
        "assumptions": params.business.model_dump(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    evaluation_plot(y, raw, cal, out_dir / "evaluation.png")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="final_evaluation"):
        mlflow.set_tags({"git_sha": git_sha(), "stage": "final_evaluation"})
        mlflow.set_tag("train_sha256", file_sha256(data_dir / TRAIN_FILE))
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifacts(str(out_dir))
        mlflow.log_artifact(str(config_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("config", "data-dir", "model-dir", "calibration-dir", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--allow-rerun", action="store_true")
    a = parser.parse_args()
    result = run(
        load_params(a.config),
        a.config,
        a.data_dir,
        a.model_dir,
        a.calibration_dir,
        a.out_dir,
        Settings(),
        a.allow_rerun,
    )
    summary = {k: round(v["value"], 4) for k, v in result["metrics"].items()}
    print(json.dumps({"threshold": result["threshold"], "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
