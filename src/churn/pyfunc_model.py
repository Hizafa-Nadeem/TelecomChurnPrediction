"""The served churn model: `score()` and the MLflow pyfunc wrapper (Step 12).

`predict()` takes the cleaned predictor columns and returns one row per customer:
`churn_probability`, `decision`, `reason_1..3`. This module is imported when a registered model is
loaded, so it only depends on what scoring needs; packaging and registration are in `churn.package`.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from mlflow.pyfunc import PythonModel

from churn.reasons import N_REASONS, contributions_of, reasons_from
from churn.validation import build_schema

ALIAS = "production"


def predictor_columns(target: str) -> list[str]:
    """Input columns of the model: the cleaned frame without the target."""
    return [c for c in build_schema(target).columns if c != target]


def score(model: Any, calibrator: Any, threshold: float, X: pd.DataFrame) -> pd.DataFrame:
    """Churn probability, decision and reasons for each row of `X` (the model's output table)."""
    features = model[:-1].transform(X)  # computed once and shared by the score and the reasons
    probability = calibrator.predict(model[-1].predict_proba(features)[:, 1])
    out = pd.DataFrame(
        {
            "churn_probability": probability,
            "decision": (probability >= threshold).astype(int),
        },
        index=X.index,
    )
    reasons = reasons_from(features, contributions_of(model, features), N_REASONS)
    return pd.concat([out, reasons], axis=1)


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
