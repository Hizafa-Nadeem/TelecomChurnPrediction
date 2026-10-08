from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.data import clean
from churn.split import load_test, load_train, make_cv, run, split_frame
from helpers import PARAMS, TARGET, make_raw


@pytest.fixture
def df() -> pd.DataFrame:
    frame = clean(make_raw(n=1000), PARAMS)
    frame[TARGET] = (np.arange(len(frame)) % 7 == 0).astype(int)  # about 14% positives
    return frame


def test_split_sizes_and_stratification(df: pd.DataFrame) -> None:
    train, test = split_frame(df, PARAMS)
    assert len(train) == 800
    assert len(test) == 200
    assert abs(train[TARGET].mean() - df[TARGET].mean()) < 0.005
    assert abs(test[TARGET].mean() - df[TARGET].mean()) < 0.005


def test_split_has_no_overlap_and_covers_all_rows(df: pd.DataFrame) -> None:
    train, test = split_frame(df, PARAMS)
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(df.index)


def test_split_is_deterministic_for_a_seed(df: pd.DataFrame) -> None:
    first, _ = split_frame(df, PARAMS)
    second, _ = split_frame(df, PARAMS)
    other, _ = split_frame(df, PARAMS.model_copy(update={"seed": PARAMS.seed + 1}))
    assert first.index.equals(second.index)
    assert not first.index.equals(other.index)


def test_cv_folds_are_stratified_and_repeatable(df: pd.DataFrame) -> None:
    train, _ = split_frame(df, PARAMS)
    y = train[TARGET]
    folds = list(make_cv(PARAMS).split(train, y))
    assert len(folds) == PARAMS.split.cv_folds * PARAMS.split.cv_repeats
    for fit_idx, val_idx in folds:
        assert set(fit_idx).isdisjoint(val_idx)
        assert abs(y.iloc[val_idx].mean() - y.mean()) < 0.01
    again = list(make_cv(PARAMS).split(train, y))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))


def test_run_writes_both_parts_and_report(df: pd.DataFrame, tmp_path: Path) -> None:
    in_dir, out_dir = tmp_path / "interim", tmp_path / "processed"
    in_dir.mkdir()
    df.to_csv(in_dir / "churn.csv", index=False)

    report = run(PARAMS, in_dir, out_dir)

    assert report["train_rows"] == 800
    assert report["test_rows"] == 200
    train, test = load_train(out_dir), load_test(out_dir)
    assert train["state"].dtype == "category"
    assert len(train) + len(test) == len(df)
