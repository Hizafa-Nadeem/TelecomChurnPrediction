"""Probability calibrator, in its own module so a saved calibrator can be loaded by any process.

A class defined in a script run as `python -m churn.calibrate` is pickled as `__main__.Calibrator`,
which no other process (the MLflow model, the API) can find; importing it from here avoids that.
"""

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("none", "sigmoid", "isotonic")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Maps raw model probabilities to calibrated ones: identity, Platt (sigmoid) or isotonic."""

    def __init__(self, method: str) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown calibration method: {method!r}")
        self.method = method
        self.model: Any = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "sigmoid":
            self.model = LogisticRegression(C=1e6).fit(_logit(p), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return np.asarray(self.model.predict_proba(_logit(p))[:, 1])
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p))
        return np.asarray(p, dtype=float)
