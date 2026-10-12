"""Package the final model as one MLflow pyfunc model and register it (Step 12).

    python -m churn.pyfunc_model --config configs/params.yaml --data-dir data/processed \
        --model-dir models --calibration-dir reports/calibration \
        --evaluation-dir reports/evaluation

The packaged model holds the feature pipeline, CatBoost, the calibrator, the decision threshold and
the reasons logic. `predict()` takes the cleaned predictor columns and returns one row per customer:
`churn_probability`, `decision`, `reason_1..3`. Only a model that passes the gate in `params.yaml`
is registered, and the registered model is checked against the local pipeline before the run ends.
"""

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

import churn
from churn.config import Params, Settings, load_params
from churn.gate import check
from churn.reasons import N_REASONS, top_reasons
from churn.split import TRAIN_FILE, load_train
from churn.train import file_sha256, git_sha
from churn.validation import CATEGORICAL_COLUMNS, build_schema

RUNTIME_PACKAGES = ["pandas", "numpy", "scikit-learn", "catboost", "pandera", "pydantic", "joblib"]
ALIAS = "production"
PARITY_ROWS = 200


def predictor_columns(target: str) -> list[str]:
    """Input columns of the model: the cleaned frame without the target."""
    return [c for c in build_schema(target).columns if c != target]


def score(model: Any, calibrator: Any, threshold: float, X: pd.DataFrame) -> pd.DataFrame:
    """Churn probability, decision and reasons for each row of `X` (the model's output table)."""
    probability = calibrator.predict(model.predict_proba(X)[:, 1])
    out = pd.DataFrame(
        {
            "churn_probability": probability,
            "decision": (probability >= threshold).astype(int),
        },
        index=X.index,
    )
    return pd.concat([out, top_reasons(model, X, N_REASONS)], axis=1)


class ChurnModel(PythonModel):
    """MLflow wrapper: loads the artifacts once and scores batches of customers."""

    def load_context(self, context: Any) -> None:
        self.model = joblib.load(context.artifacts["model"])
        self.calibrator = joblib.load(context.artifacts["calibrator"])
        spec = json.loads(Path(context.artifacts["decision"]).read_text(encoding="utf-8"))
        self.threshold: float = spec["threshold"]
        self.input_columns: list[str] = spec["input_columns"]

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        missing = [c for c in self.input_columns if c not in model_input.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        return score(self.model, self.calibrator, self.threshold, model_input[self.input_columns])


def check_parity(model_uri: str, X: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Fail unless the registered model reproduces the local pipeline's output on `X`."""
    got = mlflow.pyfunc.load_model(model_uri).predict(X)
    if not (
        got["churn_probability"].sub(expected["churn_probability"]).abs().max() < 1e-9
        and got["decision"].equals(expected["decision"])
        and got.filter(like="reason_").equals(expected.filter(like="reason_"))
    ):
        raise AssertionError("registered model output differs from the local pipeline")


def package(
    params: Params,
    data_dir: Path,
    model_dir: Path,
    calibration_dir: Path,
    evaluation_dir: Path,
    settings: Settings,
) -> str:
    """Gate, log and register the model, set the alias, verify parity; return its model URI."""
    evaluation = json.loads((evaluation_dir / "test_evaluation.json").read_text(encoding="utf-8"))
    failed = [name for name, _, _, passed in check(evaluation, params.gate) if not passed]
    if failed:
        raise RuntimeError(f"promotion gate failed, not registering: {failed}")

    target = params.data.target
    columns = predictor_columns(target)
    threshold = json.loads((calibration_dir / "calibration.json").read_text(encoding="utf-8"))[
        "decision"
    ]["threshold"]
    model = joblib.load(model_dir / "final_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")

    # Requests carry state and area_code as plain strings, so the model's input contract does too.
    X = load_train(data_dir)[columns].astype({c: str for c in CATEGORICAL_COLUMNS})
    example = X.head(PARITY_ROWS)
    expected = score(model, calibrator, threshold, example)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="package_model"), tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "decision.json"
        spec.write_text(
            json.dumps({"threshold": threshold, "input_columns": columns}), encoding="utf-8"
        )
        mlflow.set_tags(
            {
                "git_sha": git_sha(),
                "data_sha256": file_sha256(data_dir / TRAIN_FILE),
                "stage": "packaging",
            }
        )
        mlflow.log_param("threshold", threshold)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ChurnModel(),
            artifacts={
                "model": str(model_dir / "final_model.joblib"),
                "calibrator": str(model_dir / "calibrator.joblib"),
                "decision": str(spec),
            },
            code_paths=[str(Path(churn.__file__).parent)],
            pip_requirements=[f"{p}=={version(p)}" for p in RUNTIME_PACKAGES],
            signature=infer_signature(example, expected),
            input_example=example.head(5),
            registered_model_name=settings.mlflow_model_name,
        )

    new_version = info.registered_model_version
    MlflowClient().set_registered_model_alias(settings.mlflow_model_name, ALIAS, str(new_version))
    uri = f"models:/{settings.mlflow_model_name}@{ALIAS}"
    check_parity(uri, example, expected)
    return uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("config", "data-dir", "model-dir", "calibration-dir", "evaluation-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    uri = package(
        load_params(a.config),
        a.data_dir,
        a.model_dir,
        a.calibration_dir,
        a.evaluation_dir,
        Settings(),
    )
    print(f"registered and verified: {uri}")


if __name__ == "__main__":
    main()
