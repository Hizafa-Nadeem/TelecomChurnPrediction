"""Train/test split and cross-validation folds (Step 4).

    python -m churn.split --config configs/params.yaml \
        --in-dir data/interim --out-dir data/processed

The test set is written once and stays locked until the final evaluation (Step 10): only
`load_test` reads it, and only evaluate.py may call it.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from churn.config import Params, load_params
from churn.data import load_interim, read_typed_csv

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
REPORT_FILE = "split_report.json"


def split_frame(df: pd.DataFrame, params: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split with the configured seed; keeps the original row index."""
    train, test = train_test_split(
        df,
        test_size=params.split.test_size,
        stratify=df[params.data.target],
        random_state=params.seed,
    )
    return train, test


def make_cv(params: Params) -> RepeatedStratifiedKFold:
    """Repeated stratified k-fold for tuning, calibration and threshold selection."""
    return RepeatedStratifiedKFold(
        n_splits=params.split.cv_folds,
        n_repeats=params.split.cv_repeats,
        random_state=params.seed,
    )


def load_train(processed_dir: Path) -> pd.DataFrame:
    """Read the training split."""
    return read_typed_csv(processed_dir / TRAIN_FILE)


def load_test(processed_dir: Path) -> pd.DataFrame:
    """Read the locked test split. Call only from the final evaluation."""
    return read_typed_csv(processed_dir / TEST_FILE)


def run(params: Params, in_dir: Path, out_dir: Path) -> dict[str, object]:
    """Split the cleaned data, write both parts and a report. Returns the report."""
    target = params.data.target
    train, test = split_frame(load_interim(in_dir), params)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / TRAIN_FILE, index=False)
    test.to_csv(out_dir / TEST_FILE, index=False)
    report: dict[str, object] = {
        "seed": params.seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_churners": int(train[target].sum()),
        "test_churners": int(test[target].sum()),
        "train_churn_rate": round(float(train[target].mean()), 4),
        "test_churn_rate": round(float(test[target].mean()), 4),
    }
    (out_dir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_params(args.config), args.in_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
