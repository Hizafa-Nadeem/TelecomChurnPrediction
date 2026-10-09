import json
import re
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import churn
from churn.calibrate import Calibrator
from churn.config import Settings
from churn.evaluate import (
    RESULT_FILE,
    compute_metrics,
    run,
    segment_table,
    state_groups,
    wilson_interval,
)
from churn.gate import check
from churn.split import TEST_FILE, TRAIN_FILE
from churn.train import CATBOOST_FIT_PARAMS, build_models
from helpers import PARAMS, TARGET, signal_frame


def test_wilson_interval_behaves() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(50, 100)
    assert 0 <= low < 0.5 < high <= 1
    assert wilson_interval(5, 5)[1] <= 1.0


def test_metrics_on_a_hand_checked_case() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    raw = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    flagged = raw >= 0.5  # catches both churners and one stayer
    clv = np.full(10, 100.0)
    m = compute_metrics(y, raw, raw, flagged, clv, PARAMS, 0.5)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["recall"] == 1.0
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["flagged_share"] == pytest.approx(0.3)
    business = PARAMS.business
    saved = 2 * business.acceptance_rate * 100
    assert m["retention_value_per_customer"] == pytest.approx(
        (saved - 3 * business.offer_cost) / 10
    )


def test_state_groups_use_ten_high_and_ten_low_states() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "state": pd.Categorical(rng.integers(0, 51, 5000).astype(str)),
            TARGET: rng.integers(0, 2, 5000),
        }
    )
    groups = state_groups(train, TARGET)
    values = list(groups.values())
    assert values.count("high-churn states") == 10
    assert values.count("low-churn states") == 10


def test_segment_table_covers_every_customer() -> None:
    df = signal_frame(300)
    X, y = df.drop(columns=TARGET), df[TARGET].to_numpy()
    flagged = np.random.default_rng(0).uniform(size=len(y)) > 0.7
    groups = {s: "middle" for s in X["state"].astype(str).unique()}
    rows = segment_table(X, y, flagged, groups)
    for segment in {r["segment"] for r in rows}:
        assert sum(r["n"] for r in rows if r["segment"] == segment) == len(y)
    assert all(0 <= r["recall"] <= 1 and 0 <= r["precision"] <= 1 for r in rows)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Train a tiny model, calibrator and threshold file on synthetic data."""
    data_dir, model_dir, cal_dir = tmp_path / "processed", tmp_path / "models", tmp_path / "cal"
    for d in (data_dir, model_dir, cal_dir):
        d.mkdir()
    train, test = signal_frame(400), signal_frame(300, seed=1)
    train.to_csv(data_dir / TRAIN_FILE, index=False)
    test.to_csv(data_dir / TEST_FILE, index=False)
    X, y = train.drop(columns=TARGET), train[TARGET]
    model = clone(build_models(PARAMS)["catboost"]).set_params(model__iterations=20)
    model.fit(X, y, **CATBOOST_FIT_PARAMS)
    calibrator = Calibrator("isotonic").fit(model.predict_proba(X)[:, 1], y.to_numpy())
    joblib.dump(model, model_dir / "final_model.joblib")
    joblib.dump(calibrator, model_dir / "calibrator.joblib")
    (cal_dir / "calibration.json").write_text(
        json.dumps({"decision": {"threshold": 0.3}}), encoding="utf-8"
    )
    return data_dir, model_dir, cal_dir


def test_run_evaluates_once_and_refuses_a_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("churn.evaluate.N_BOOTSTRAP", 30)
    data_dir, model_dir, cal_dir = _artifacts(tmp_path)
    out_dir, config = tmp_path / "evaluation", tmp_path / "params.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")  # only logged as an artifact
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    settings = Settings(mlflow_tracking_uri=mlflow.get_tracking_uri(), mlflow_experiment_name="t")
    args = (PARAMS, config, data_dir, model_dir, cal_dir, out_dir, settings)

    result = run(*args)

    assert result["n_test"] == 300
    assert result["confusion"]["tp"] + result["confusion"]["fn"] == result["churners"]
    low, high = result["metrics"]["pr_auc"]["ci95"]
    assert low <= high
    assert (out_dir / RESULT_FILE).exists()
    assert (out_dir / "evaluation.png").exists()
    with pytest.raises(FileExistsError):
        run(*args)
    assert run(*args, allow_rerun=True)["n_test"] == 300


def test_only_the_final_evaluation_reads_the_test_split() -> None:
    src = Path(churn.__file__).parent
    users = {
        f.name
        for f in src.glob("*.py")
        if re.search(r"\bload_test\b|test\.csv", f.read_text("utf-8"))
    }
    assert users <= {"split.py", "evaluate.py"}


def test_gate_passes_and_fails_on_the_fixed_criteria() -> None:
    def result(pr: float, roc: float, recall: float, logreg: float) -> dict[str, Any]:
        return {
            "metrics": {
                "pr_auc": {"value": pr},
                "roc_auc": {"value": roc},
                "recall": {"value": recall},
            },
            "baselines": {"logreg_pr_auc": logreg},
        }

    good = check(result(0.90, 0.93, 0.86, 0.55), PARAMS.gate)
    assert all(passed for *_, passed in good)
    bad = check(result(0.80, 0.93, 0.86, 0.55), PARAMS.gate)
    assert [name for name, *_, passed in bad if not passed] == ["PR-AUC"]
    worse_than_baseline = check(result(0.90, 0.93, 0.86, 0.95), PARAMS.gate)
    assert [name for name, *_, passed in worse_than_baseline if not passed] == [
        "PR-AUC vs logistic regression"
    ]
