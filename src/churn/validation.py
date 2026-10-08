"""Data contract checks: pandera schema and the leakage check (Step 2)."""

import pandas as pd
import pandera.pandas as pa
from pandas.api.types import is_float_dtype, is_numeric_dtype
from pandera import Check

CATEGORICAL_COLUMNS = ["state", "area_code"]
BINARY_COLUMNS = ["international_plan", "voice_mail_plan"]
USAGE_PERIODS = ["day", "eve", "night", "intl"]


class LeakageError(ValueError):
    """Raised when a feature is identifier-like or separates the target perfectly."""


def build_schema(target: str) -> pa.DataFrameSchema:
    """Schema for the cleaned frame (identifier already dropped)."""
    columns: dict[str, pa.Column] = {
        "state": pa.Column("category", Check.isin([str(i) for i in range(51)])),
        "area_code": pa.Column("category", Check.isin(["408", "415", "510"])),
        "account_length": pa.Column(int, Check.ge(1)),
        "number_vmail_messages": pa.Column(int, Check.ge(0)),
        "number_customer_service_calls": pa.Column(int, Check.in_range(0, 9)),
        target: pa.Column(int, Check.isin([0, 1])),
    }
    for flag in BINARY_COLUMNS:
        columns[flag] = pa.Column(int, Check.isin([0, 1]))
    for period in USAGE_PERIODS:
        columns[f"total_{period}_minutes"] = pa.Column(float, Check.ge(0))
        columns[f"total_{period}_calls"] = pa.Column(int, Check.ge(0))
        columns[f"total_{period}_charge"] = pa.Column(float, Check.ge(0))
    # nullable=False is the pandera default, so missing values fail the schema.
    return pa.DataFrameSchema(columns, strict=True, unique=list(columns), report_duplicates="all")


def check_leakage(df: pd.DataFrame, target: str) -> None:
    """Fail if a feature is an identifier or separates the target perfectly.

    Identifier-like means unique per row in an integer or text column; continuous floats can be
    unique by chance, so they are only tested for perfect separation.
    """
    y = df[target]
    problems: list[str] = []
    for col in df.columns.drop(target):
        x = df[col]
        if not is_float_dtype(x) and x.nunique() == len(df):
            problems.append(f"{col}: unique per row (identifier-like)")
        elif is_numeric_dtype(x):
            lo, hi = x[y == 0], x[y == 1]
            if lo.max() < hi.min() or hi.max() < lo.min():
                problems.append(f"{col}: separates the target perfectly")
        elif (df.groupby(col, observed=True)[target].nunique() == 1).all():
            problems.append(f"{col}: every level has a single target value")
    if problems:
        raise LeakageError("; ".join(problems))
