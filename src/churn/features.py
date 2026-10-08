"""Feature engineering shared by training and serving (Step 5).

`derive_features` is stateless, so it cannot leak between train and test. The pipelines take the
cleaned frame without the target; any extra column (including the target) is dropped because the
output is built from a fixed column list.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.validation import BINARY_COLUMNS, CATEGORICAL_COLUMNS, USAGE_PERIODS

HIGH_SERVICE_CALLS = 4
# About the top quartile of training day minutes (216); EDA shows churn is concentrated there
# for customers without voicemail. Fixed so the transform stays stateless.
HEAVY_DAY_MINUTES = 215
RARE_LEVEL_FREQUENCY = 0.02

FLAG_COLUMNS = [*BINARY_COLUMNS, "high_service_calls", "heavy_day_no_voicemail"]
NUMERIC_COLUMNS = [
    "account_length",
    "number_vmail_messages",
    "number_customer_service_calls",
    *[f"total_{p}_{kind}" for p in USAGE_PERIODS for kind in ("minutes", "calls")],
    "total_minutes",
    "total_calls",
    "total_charge",
    "avg_day_call_length",
    "avg_eve_call_length",
    "avg_night_call_length",
    "intl_minutes_per_call",
    "day_minutes_share",
]
FEATURE_COLUMNS = [*CATEGORICAL_COLUMNS, *FLAG_COLUMNS, *NUMERIC_COLUMNS]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, with 0 where den is 0 so the result is always finite."""
    safe = np.divide(
        num.to_numpy(float), den.to_numpy(float), out=np.zeros(len(num)), where=den.to_numpy() > 0
    )
    return pd.Series(safe, index=num.index)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features, drop the four `total_*_charge` columns and return `FEATURE_COLUMNS`."""
    out = df.copy()
    minutes = {p: out[f"total_{p}_minutes"] for p in USAGE_PERIODS}
    calls = {p: out[f"total_{p}_calls"] for p in USAGE_PERIODS}
    out["total_minutes"] = sum(minutes.values())
    out["total_calls"] = sum(calls.values())
    out["total_charge"] = sum(out[f"total_{p}_charge"] for p in USAGE_PERIODS)
    for p in ("day", "eve", "night"):
        out[f"avg_{p}_call_length"] = _ratio(minutes[p], calls[p])
    out["intl_minutes_per_call"] = _ratio(minutes["intl"], calls["intl"])
    out["day_minutes_share"] = _ratio(minutes["day"], out["total_minutes"])
    out["high_service_calls"] = (out["number_customer_service_calls"] >= HIGH_SERVICE_CALLS).astype(
        int
    )
    out["heavy_day_no_voicemail"] = (
        (out["total_day_minutes"] >= HEAVY_DAY_MINUTES) & (out["voice_mail_plan"] == 0)
    ).astype(int)
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str)
    return out[FEATURE_COLUMNS]


def build_pipeline(kind: str) -> Pipeline:
    """Feature pipeline for `kind` in {"catboost", "logreg"}; fit on training data only."""
    derive = FunctionTransformer(derive_features).set_output(transform="pandas")
    if kind == "catboost":
        return Pipeline([("derive", derive)])
    if kind == "logreg":
        onehot = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=RARE_LEVEL_FREQUENCY,
            sparse_output=False,
        )
        encode = ColumnTransformer(
            [
                ("onehot", onehot, CATEGORICAL_COLUMNS),
                ("scale", StandardScaler(), NUMERIC_COLUMNS),
                ("flags", "passthrough", FLAG_COLUMNS),
            ]
        ).set_output(transform="pandas")
        return Pipeline([("derive", derive), ("encode", encode)])
    raise ValueError(f"unknown pipeline kind: {kind!r}")


def catboost_cat_features() -> list[str]:
    """Categorical column names to pass to CatBoost as `cat_features`."""
    return list(CATEGORICAL_COLUMNS)
