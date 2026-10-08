"""Ingestion and validation stage (Steps 1-2).

    python -m churn.data --config configs/params.yaml --raw-dir data/raw --out-dir data/interim

Downloads the raw file once (never overwritten), drops the identifier, casts categoricals,
validates against the data contract and writes the cleaned frame plus a report.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from churn.config import Params, load_params
from churn.validation import CATEGORICAL_COLUMNS, build_schema, check_leakage

RAW_FILE = "churn.csv"
INTERIM_FILE = "churn.csv"
REPORT_FILE = "validation_report.json"


def fetch_raw(raw_dir: Path, openml_id: int) -> Path:
    """Download the OpenML dataset to `raw_dir` unless it is already there."""
    path = raw_dir / RAW_FILE
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        frame = fetch_openml(data_id=openml_id, as_frame=True, parser="auto").frame
        frame.to_csv(path, index=False)
    return path


def clean(raw: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Drop the identifier and cast `state` and `area_code` to categorical strings."""
    df = raw.drop(columns=[params.data.id_column])
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).astype("category")
    return df


def load_interim(out_dir: Path) -> pd.DataFrame:
    """Read the cleaned frame back with `state` and `area_code` as categorical strings."""
    df = pd.read_csv(out_dir / INTERIM_FILE, dtype={c: str for c in CATEGORICAL_COLUMNS})
    return df.astype({c: "category" for c in CATEGORICAL_COLUMNS})


def run(params: Params, raw_dir: Path, out_dir: Path) -> dict[str, object]:
    """Fetch, clean and validate; write the cleaned data and a report. Returns the report."""
    raw = pd.read_csv(fetch_raw(raw_dir, params.data.openml_id))
    df = clean(raw, params)
    build_schema(params.data.target).validate(df, lazy=True)
    check_leakage(df, params.data.target)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / INTERIM_FILE, index=False)
    report: dict[str, object] = {
        "rows": len(df),
        "columns": list(df.columns),
        "churn_rate": round(float(df[params.data.target].mean()), 4),
        "dropped": {params.data.id_column: "identifier, never a feature"},
        "checks": ["schema", "no_missing", "no_duplicates", "leakage"],
    }
    (out_dir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(load_params(args.config), args.raw_dir, args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
