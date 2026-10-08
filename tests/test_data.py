from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from churn.config import load_params
from churn.data import RAW_FILE, clean, load_interim, run
from churn.validation import LeakageError, build_schema, check_leakage

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


@pytest.fixture
def df() -> pd.DataFrame:
    return clean(make_raw(), PARAMS)


def test_clean_drops_identifier_and_casts_categoricals(df: pd.DataFrame) -> None:
    assert PARAMS.data.id_column not in df.columns
    assert df["state"].dtype == "category"
    assert set(df["area_code"].cat.categories) <= {"408", "415", "510"}


def test_valid_frame_passes_schema_and_leakage(df: pd.DataFrame) -> None:
    build_schema(TARGET).validate(df, lazy=True)
    check_leakage(df, TARGET)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("number_customer_service_calls", 10),
        ("international_plan", 2),
        ("total_day_minutes", -1.0),
        ("account_length", 0),
        (TARGET, 2),
    ],
)
def test_out_of_range_value_fails_schema(df: pd.DataFrame, column: str, value: float) -> None:
    df.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaErrors):
        build_schema(TARGET).validate(df, lazy=True)


def test_missing_value_fails_schema(df: pd.DataFrame) -> None:
    df["total_day_minutes"] = df["total_day_minutes"].astype(float)
    df.loc[0, "total_day_minutes"] = np.nan
    with pytest.raises(pandera.errors.SchemaErrors):
        build_schema(TARGET).validate(df, lazy=True)


def test_duplicate_rows_fail_schema(df: pd.DataFrame) -> None:
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaErrors):
        build_schema(TARGET).validate(df, lazy=True)


def test_leakage_detects_identifier_like_column(df: pd.DataFrame) -> None:
    df["row_id"] = np.arange(len(df))
    with pytest.raises(LeakageError, match="row_id"):
        check_leakage(df, TARGET)


def test_leakage_detects_perfectly_separating_feature(df: pd.DataFrame) -> None:
    df["total_day_minutes"] = df[TARGET] * 1000.0 + np.random.default_rng(1).uniform(0, 1, len(df))
    with pytest.raises(LeakageError, match="total_day_minutes"):
        check_leakage(df, TARGET)


def test_run_writes_clean_data_and_report(tmp_path: Path) -> None:
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "interim"
    raw_dir.mkdir()
    make_raw().to_csv(raw_dir / RAW_FILE, index=False)  # present, so no download happens

    report = run(PARAMS, raw_dir, out_dir)

    assert report["rows"] == 200
    assert (out_dir / "validation_report.json").exists()
    back = load_interim(out_dir)
    assert back["state"].dtype == "category"
    assert PARAMS.data.id_column not in back.columns
