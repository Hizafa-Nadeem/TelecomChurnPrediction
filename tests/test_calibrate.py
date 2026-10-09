import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from churn.calibrate import (
    Calibrator,
    choose_threshold,
    cross_fitted,
    retention_value,
    run,
)
from churn.config import BusinessParams, Settings
from churn.split import TRAIN_FILE
from helpers import PARAMS, signal_frame

BUSINESS = BusinessParams(remaining_months=12, acceptance_rate=0.5, offer_cost=10)


def miscalibrated(n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True churn probability p, outcomes drawn from it, and a distorted score p**3."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    return p**3, (rng.uniform(size=n) < p).astype(int)


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError):
        Calibrator("platt")


def test_none_is_identity() -> None:
    p = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(Calibrator("none").fit(p, np.array([0, 1, 1])).predict(p), p)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrators_fix_a_distorted_score(method: str) -> None:
    score, y = miscalibrated()
    fixed = cross_fitted(method, score, y, seed=1)
    assert brier_score_loss(y, fixed) < brier_score_loss(y, score)


def test_isotonic_is_monotone_and_bounded() -> None:
    score, y = miscalibrated()
    grid = np.linspace(0, 1, 200)
    out = Calibrator("isotonic").fit(score, y).predict(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0
    assert out.max() <= 1


def test_retention_value_on_a_hand_checked_case() -> None:
    p = np.array([0.9, 0.2, 0.6])
    y = np.array([1, 0, 1])
    clv = np.array([100.0, 100.0, 100.0])
    # Flagged at 0.5: customers 0 and 2, both churners: 2 x (0.5 x 100) - 2 x 10 = 80.
    assert retention_value(p, y, clv, BUSINESS, 0.5) == pytest.approx(80.0)
    assert retention_value(p, y, clv, BUSINESS, 0.95) == 0.0


def test_threshold_lands_near_break_even_for_calibrated_probabilities() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    clv = np.full(p.size, 100.0)  # break-even = 10 / (0.5 x 100) = 0.2
    result = choose_threshold(p, y, clv, BUSINESS)
    assert result["break_even_probability"] == pytest.approx(0.2)
    assert abs(result["threshold"] - 0.2) < 0.05
    assert result["value"] > 0
    assert result["value"] > result["value_flag_everyone"]


def test_flat_value_curve_picks_the_threshold_closest_to_break_even() -> None:
    p = np.array([0.01] * 90 + [0.99] * 10)  # nothing between: every threshold in between ties
    y = np.array([0] * 90 + [1] * 10)
    clv = np.full(100, 1000.0)
    result = choose_threshold(p, y, clv, BUSINESS)
    assert result["threshold"] == pytest.approx(result["break_even_probability"], abs=0.01)


def test_run_saves_model_calibrator_threshold_and_plots(tmp_path: Path) -> None:
    data_dir, tuning_dir = tmp_path / "processed", tmp_path / "tuning"
    out_dir, model_dir = tmp_path / "calibration", tmp_path / "models"
    data_dir.mkdir()
    tuning_dir.mkdir()
    frame = signal_frame(400)
    frame.to_csv(data_dir / TRAIN_FILE, index=False)
    best = {"depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "iterations": 20}
    (tuning_dir / "best_params.json").write_text(
        json.dumps({"best_params": best}), encoding="utf-8"
    )
    config = tmp_path / "params.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")  # only logged as an artifact
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    settings = Settings(mlflow_tracking_uri=mlflow.get_tracking_uri(), mlflow_experiment_name="t")
    fast = PARAMS.model_copy(
        update={"split": PARAMS.split.model_copy(update={"cv_folds": 3, "cv_repeats": 1})}
    )

    result = run(fast, config, data_dir, tuning_dir, out_dir, model_dir, settings)

    assert 0 < result["decision"]["threshold"] < 1
    for name in ("calibration.json", "reliability.png", "threshold_value.png"):
        assert (out_dir / name).exists()
    model = joblib.load(model_dir / "final_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    raw = model.predict_proba(frame.drop(columns="class").head(5))[:, 1]
    assert ((calibrator.predict(raw) >= 0) & (calibrator.predict(raw) <= 1)).all()
