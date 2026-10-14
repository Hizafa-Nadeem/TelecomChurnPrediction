"""Per-customer churn reasons from SHAP values (Step 11), light enough for the served model.

SHAP values come from CatBoost's exact TreeSHAP (`ShapValues`), so categorical columns need no
special handling and the contributions add up to the model's raw score.
"""

from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from churn.features import catboost_cat_features

N_REASONS = 3
# Smaller pushes (about +10% on the odds of churn) are noise, not reasons to act on.
MIN_CONTRIBUTION = 0.1


def shap_contributions(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Engineered features and their SHAP contributions to the log-odds of churn.

    `model` is the fitted pipeline (feature step, then CatBoost). Row i's contributions plus the
    model's base value equal its raw score.
    """
    features = model[:-1].transform(X)
    return features, contributions_of(model, features)


def contributions_of(model: Any, features: pd.DataFrame) -> np.ndarray:
    """SHAP contributions for already-engineered `features` (saves recomputing them)."""
    pool = Pool(features, cat_features=catboost_cat_features())
    return np.asarray(model[-1].get_feature_importance(pool, type="ShapValues")[:, :-1])


def _describe(name: str, value: object) -> str:
    return f"{name}={value:.1f}" if isinstance(value, float) else f"{name}={value}"


def top_reasons(model: Any, X: pd.DataFrame, k: int = N_REASONS) -> pd.DataFrame:
    """The `k` features pushing each customer's churn risk up the most, as `reason_1..k`.

    Only features whose contribution is at least `MIN_CONTRIBUTION` are listed, so customers with
    no clear risk driver get empty slots.
    """
    features, contributions = shap_contributions(model, X)
    return reasons_from(features, contributions, k)


def reasons_from(
    features: pd.DataFrame, contributions: np.ndarray, k: int = N_REASONS
) -> pd.DataFrame:
    """Format the top `k` risk-raising features per row from precomputed contributions."""
    rows = []
    for values, contrib in zip(features.to_numpy(dtype=object), contributions, strict=True):
        best = np.argsort(-contrib)[:k]
        rows.append(
            [
                _describe(features.columns[i], values[i]) if contrib[i] >= MIN_CONTRIBUTION else ""
                for i in best
            ]
        )
    return pd.DataFrame(rows, columns=[f"reason_{i + 1}" for i in range(k)], index=features.index)
