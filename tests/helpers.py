"""Test helpers shared across test modules."""

from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import load_params
from churn.data import clean

PARAMS = load_params(Path(__file__).resolve().parents[1] / "configs" / "params.yaml")
TARGET = PARAMS.data.target


def make_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame shaped like the OpenML file, with no signal in any feature."""
    rng = np.random.default_rng(seed)
    raw: dict[str, object] = {
        "state": rng.integers(0, 3, n),
        "account_length": rng.integers(1, 240, n),
        "area_code": rng.choice([408, 415, 510], n),
        "phone_number": np.arange(n),
        "international_plan": rng.integers(0, 2, n),
        "voice_mail_plan": rng.integers(0, 2, n),
        "number_vmail_messages": rng.integers(0, 50, n),
        "number_customer_service_calls": rng.integers(0, 10, n),
    }
    for period in ["day", "eve", "night", "intl"]:
        raw[f"total_{period}_minutes"] = rng.uniform(0, 300, n).round(2)
        raw[f"total_{period}_calls"] = rng.integers(0, 150, n)
        raw[f"total_{period}_charge"] = rng.uniform(0, 50, n).round(2)
    raw[TARGET] = rng.integers(0, 2, n)
    return pd.DataFrame(raw)


def signal_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic data where churn depends on international plan and 4+ service calls."""
    df = clean(make_raw(n, seed), PARAMS)
    rng = np.random.default_rng(seed)
    logit = -2.5 + 2.0 * df["international_plan"] + 2.0 * (df["number_customer_service_calls"] >= 4)
    df[TARGET] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
