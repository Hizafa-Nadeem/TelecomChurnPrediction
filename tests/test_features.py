"""Tests for the shared feature pipeline (Step 5)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.data import clean
from churn.features import FEATURE_COLUMNS, build_pipeline, catboost_cat_features, derive_features
from helpers import PARAMS, TARGET, make_raw


def cleaned(n: int = 200, seed: int = 0) -> pd.DataFrame:
    return clean(make_raw(n, seed), PARAMS)


def one_row(**overrides: float) -> pd.DataFrame:
    row = cleaned(5).iloc[[0]].drop(columns=TARGET).copy()
    for col, value in overrides.items():
        row[col] = value
    return row


def test_derived_values_on_hand_made_row() -> None:
    row = one_row(
        total_day_minutes=100.0, total_day_calls=50, total_day_charge=17.0,
        total_eve_minutes=60.0, total_eve_calls=20, total_eve_charge=5.0,
        total_night_minutes=40.0, total_night_calls=10, total_night_charge=2.0,
        total_intl_minutes=10.0, total_intl_calls=5, total_intl_charge=3.0,
        number_customer_service_calls=4, voice_mail_plan=0,
    )  # fmt: skip
    out = derive_features(row).iloc[0]
    assert out["total_minutes"] == 210
    assert out["total_calls"] == 85
    assert out["total_charge"] == pytest.approx(27.0)
    assert out["avg_day_call_length"] == 2.0
    assert out["avg_eve_call_length"] == 3.0
    assert out["avg_night_call_length"] == 4.0
    assert out["intl_minutes_per_call"] == 2.0
    assert out["day_minutes_share"] == pytest.approx(100 / 210)
    assert out["high_service_calls"] == 1
    assert out["heavy_day_no_voicemail"] == 0  # 100 day minutes is below the cutoff


@pytest.mark.parametrize(
    ("minutes", "voicemail", "expected"),
    [(215.0, 0, 1), (214.9, 0, 0), (300.0, 1, 0)],
)
def test_heavy_day_no_voicemail_flag(minutes: float, voicemail: int, expected: int) -> None:
    row = one_row(total_day_minutes=minutes, voice_mail_plan=voicemail)
    assert derive_features(row).iloc[0]["heavy_day_no_voicemail"] == expected


def test_dropped_features_are_absent() -> None:
    cols = derive_features(cleaned()).columns
    assert "account_length_bucket" not in cols
    assert "has_voicemail_usage" not in cols


def test_charge_columns_dropped() -> None:
    out = derive_features(cleaned())
    assert [c for c in out.columns if c.endswith("_charge")] == ["total_charge"]


def test_zero_calls_give_finite_values() -> None:
    row = one_row(total_day_calls=0, total_eve_calls=0, total_night_calls=0, total_intl_calls=0)
    row[[c for c in row.columns if c.endswith("_minutes")]] = 0.0
    out = derive_features(row).select_dtypes("number")
    assert np.isfinite(out.to_numpy()).all()


def test_identical_columns_across_frames() -> None:
    frames = (cleaned(50, 1), cleaned(7, 2), one_row())
    assert all(list(derive_features(f).columns) == FEATURE_COLUMNS for f in frames)


def test_target_column_is_ignored() -> None:
    assert TARGET not in derive_features(cleaned()).columns


def test_logreg_pipeline_feeds_logistic_regression() -> None:
    df = cleaned(300)
    X, y = df.drop(columns=TARGET), df[TARGET]
    pipe = build_pipeline("logreg")
    Xt = pipe.fit_transform(X)
    assert Xt.select_dtypes(exclude="number").empty
    proba = LogisticRegression(max_iter=500).fit(Xt, y).predict_proba(pipe.transform(one_row()))
    assert ((proba >= 0) & (proba <= 1)).all()


def test_catboost_pipeline_keeps_string_categoricals() -> None:
    out = build_pipeline("catboost").fit_transform(cleaned().drop(columns=TARGET))
    for col in catboost_cat_features():
        assert out[col].map(type).eq(str).all()


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError):
        build_pipeline("xgboost")
