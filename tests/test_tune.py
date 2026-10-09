import json
from pathlib import Path

import mlflow
import optuna
import pytest

from churn.config import Settings
from churn.split import TRAIN_FILE
from churn.tune import BEST_PARAMS_FILE, run, suggest_params, tune
from helpers import PARAMS, TARGET, signal_frame

FAST = PARAMS.model_copy(
    update={
        "split": PARAMS.split.model_copy(update={"cv_folds": 3, "cv_repeats": 1}),
        "tune": PARAMS.tune.model_copy(update={"n_trials": 2, "cv_folds": 2, "cv_repeats": 1}),
    }
)


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")


@pytest.fixture
def fast_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same parameters, but tiny `iterations`, so tests train in seconds."""

    def small(trial: optuna.Trial) -> dict[str, float | int]:
        return {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "iterations": trial.suggest_int("iterations", 10, 30, step=10),
        }

    monkeypatch.setattr("churn.tune.suggest_params", small)


def test_search_space_stays_in_bounds() -> None:
    seen: list[dict[str, float | int]] = []

    def probe(trial: optuna.Trial) -> float:
        seen.append(suggest_params(trial))
        return 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.create_study().optimize(probe, n_trials=25)
    assert all(4 <= p["depth"] <= 8 for p in seen)
    assert all(0.02 <= p["learning_rate"] <= 0.3 for p in seen)
    assert all(1 <= p["l2_leaf_reg"] <= 30 for p in seen)
    assert all(200 <= p["iterations"] <= 800 and p["iterations"] % 100 == 0 for p in seen)


def test_same_seed_gives_the_same_best_params(fast_search: None) -> None:
    df = signal_frame(300)
    X, y = df.drop(columns=TARGET), df[TARGET]
    first, second = tune(X, y, FAST), tune(X, y, FAST)
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value
    assert 0 <= first.best_value <= 1


def test_run_writes_best_params_and_logs_a_summary_run(tmp_path: Path, fast_search: None) -> None:
    data_dir, out_dir = tmp_path / "processed", tmp_path / "tuning"
    data_dir.mkdir()
    signal_frame(300).to_csv(data_dir / TRAIN_FILE, index=False)
    config = tmp_path / "params.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")  # only logged as an artifact
    settings = Settings(
        mlflow_tracking_uri=mlflow.get_tracking_uri(), mlflow_experiment_name="test"
    )

    result = run(FAST, config, data_dir, out_dir, settings)

    saved = json.loads((out_dir / BEST_PARAMS_FILE).read_text(encoding="utf-8"))
    assert saved["best_params"] == result["best_params"]
    assert set(saved["best_params"]) == {"depth", "learning_rate", "l2_leaf_reg", "iterations"}
    assert saved["trials"] == 2
