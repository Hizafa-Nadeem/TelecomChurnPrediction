from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from churn.calibrator import Calibrator
from churn.package import package
from churn.pyfunc_model import ALIAS, predictor_columns, score
from churn.split import TRAIN_FILE
from helpers import MODEL_NAME, PARAMS, TARGET, evaluation_json, registry_settings


def test_calibrator_is_importable_from_its_own_module() -> None:
    # A class defined in a `python -m` entry script is pickled as __main__.X and cannot be loaded
    # by the served model; the calibrator must live in an importable module.
    assert Calibrator.__module__ == "churn.calibrator"


def test_input_columns_are_the_19_predictors() -> None:
    columns = predictor_columns(TARGET)
    assert len(columns) == 19
    assert TARGET not in columns
    assert "phone_number" not in columns


def test_packaged_model_matches_the_local_pipeline(workspace: dict[str, Path]) -> None:
    uri = package(
        PARAMS,
        workspace["processed"],
        workspace["models"],
        workspace["calibration"],
        workspace["evaluation"],
        registry_settings(),
    )

    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    assert uri == f"models:/{MODEL_NAME}@{ALIAS}"
    assert int(version.version) == 1

    model = joblib.load(workspace["models"] / "final_model.joblib")
    calibrator = joblib.load(workspace["models"] / "calibrator.joblib")
    X = pd.read_csv(workspace["processed"] / TRAIN_FILE, dtype={"state": str, "area_code": str})
    X = X[predictor_columns(TARGET)].head(30)
    expected = score(model, calibrator, 0.3, X)
    got = mlflow.pyfunc.load_model(uri).predict(X)

    assert list(got.columns) == [
        "churn_probability",
        "decision",
        "reason_1",
        "reason_2",
        "reason_3",
    ]
    assert np.allclose(got["churn_probability"], expected["churn_probability"], atol=1e-9)
    assert got["decision"].equals(expected["decision"])
    assert got["decision"].isin([0, 1]).all()
    assert ((got["churn_probability"] >= 0.3).astype(int) == got["decision"]).all()
    assert got[["reason_1", "reason_2", "reason_3"]].equals(
        expected[["reason_1", "reason_2", "reason_3"]]
    )


def test_a_missing_input_column_is_rejected(workspace: dict[str, Path]) -> None:
    uri = package(
        PARAMS,
        workspace["processed"],
        workspace["models"],
        workspace["calibration"],
        workspace["evaluation"],
        registry_settings(),
    )
    X = pd.read_csv(workspace["processed"] / TRAIN_FILE, dtype={"state": str, "area_code": str})
    X = X[predictor_columns(TARGET)].head(3).drop(columns="total_day_minutes")
    with pytest.raises(Exception, match="missing"):
        mlflow.pyfunc.load_model(uri).predict(X)


def test_a_model_that_fails_the_gate_is_not_registered(workspace: dict[str, Path]) -> None:
    (workspace["evaluation"] / "test_evaluation.json").write_text(
        evaluation_json(pr_auc=0.70), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="promotion gate"):
        package(
            PARAMS,
            workspace["processed"],
            workspace["models"],
            workspace["calibration"],
            workspace["evaluation"],
            registry_settings(),
        )
    assert MlflowClient().search_registered_models() == []
