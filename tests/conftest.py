from pathlib import Path

import joblib
import mlflow
import pytest
from sklearn.base import clone

from churn.calibrator import Calibrator
from churn.split import TRAIN_FILE
from churn.train import CATBOOST_FIT_PARAMS, build_models
from helpers import PARAMS, TARGET, evaluation_json, signal_frame


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """A tiny trained model, calibrator, threshold and a passing evaluation, plus local MLflow."""
    dirs = {n: tmp_path / n for n in ("processed", "models", "calibration", "evaluation")}
    for d in dirs.values():
        d.mkdir()
    frame = signal_frame(400)
    frame.to_csv(dirs["processed"] / TRAIN_FILE, index=False)
    X, y = frame.drop(columns=TARGET), frame[TARGET]
    model = clone(build_models(PARAMS)["catboost"]).set_params(model__iterations=30)
    model.fit(X, y, **CATBOOST_FIT_PARAMS)
    calibrator = Calibrator("isotonic").fit(model.predict_proba(X)[:, 1], y.to_numpy())
    joblib.dump(model, dirs["models"] / "final_model.joblib")
    joblib.dump(calibrator, dirs["models"] / "calibrator.joblib")
    (dirs["calibration"] / "calibration.json").write_text(
        '{"decision": {"threshold": 0.3}}', encoding="utf-8"
    )
    (dirs["evaluation"] / "test_evaluation.json").write_text(evaluation_json(), encoding="utf-8")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("t", artifact_location=(tmp_path / "artifacts").as_uri())
    return dirs
