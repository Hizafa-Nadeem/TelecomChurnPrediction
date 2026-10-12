from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from catboost import Pool
from sklearn.base import clone

from churn.config import Settings
from churn.explain import run
from churn.features import catboost_cat_features
from churn.reasons import MIN_CONTRIBUTION, shap_contributions, top_reasons
from churn.split import TRAIN_FILE
from churn.train import CATBOOST_FIT_PARAMS, build_models
from helpers import PARAMS, TARGET, signal_frame


@pytest.fixture(scope="module")
def fitted() -> tuple[object, pd.DataFrame]:
    """A small model on synthetic data where churn depends on the plan and on 4+ service calls."""
    df = signal_frame(600)
    X, y = df.drop(columns=TARGET), df[TARGET]
    model = clone(build_models(PARAMS)["catboost"]).set_params(model__iterations=80)
    model.fit(X, y, **CATBOOST_FIT_PARAMS)
    return model, X


def test_contributions_add_up_to_the_model_score(fitted: tuple[object, pd.DataFrame]) -> None:
    model, X = fitted
    features, contributions = shap_contributions(model, X.head(50))
    assert contributions.shape == features.shape
    pool = Pool(features, cat_features=catboost_cat_features())
    raw = model[-1].predict(pool, prediction_type="RawFormulaVal")  # type: ignore[index]
    # The base value is the same for every row, so differences must match exactly.
    assert np.allclose(np.diff(contributions.sum(axis=1)), np.diff(raw), atol=1e-6)


def test_reasons_have_the_right_shape_and_only_real_drivers(
    fitted: tuple[object, pd.DataFrame],
) -> None:
    model, X = fitted
    reasons = top_reasons(model, X.head(40))
    assert list(reasons.columns) == ["reason_1", "reason_2", "reason_3"]
    assert reasons.index.equals(X.head(40).index)
    _, contributions = shap_contributions(model, X.head(40))
    listed = (reasons != "").sum(axis=1).to_numpy()
    assert np.array_equal(listed, np.minimum((contributions >= MIN_CONTRIBUTION).sum(axis=1), 3))


def test_a_high_risk_customer_gets_the_true_drivers(fitted: tuple[object, pd.DataFrame]) -> None:
    model, X = fitted
    row = X.iloc[[0]].copy()
    row["international_plan"] = 1
    row["number_customer_service_calls"] = 6
    reasons = " ".join(top_reasons(model, row).iloc[0])
    assert "international_plan=1" in reasons or "number_customer_service_calls=6" in reasons


def test_absent_risk_drivers_are_not_named_as_reasons(
    fitted: tuple[object, pd.DataFrame],
) -> None:
    model, X = fitted
    row = X.iloc[[0]].copy()
    row["international_plan"] = 0
    row["number_customer_service_calls"] = 0
    row["high_service_calls"] = 0
    reasons = " ".join(top_reasons(model, row).iloc[0])
    assert "international_plan" not in reasons
    assert "service_calls" not in reasons


def test_run_writes_importance_plots_and_examples(
    fitted: tuple[object, pd.DataFrame], tmp_path: Path
) -> None:
    model, _ = fitted
    data_dir, model_dir, out_dir = tmp_path / "processed", tmp_path / "models", tmp_path / "out"
    data_dir.mkdir()
    model_dir.mkdir()
    signal_frame(300, seed=2).to_csv(data_dir / TRAIN_FILE, index=False)
    joblib.dump(model, model_dir / "final_model.joblib")
    config = tmp_path / "params.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")  # only logged as an artifact
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    settings = Settings(mlflow_tracking_uri=mlflow.get_tracking_uri(), mlflow_experiment_name="t")

    importance = run(PARAMS, config, data_dir, model_dir, out_dir, settings)

    assert importance.is_monotonic_decreasing
    assert set(importance.index[:3]) & {
        "international_plan",
        "number_customer_service_calls",
        "high_service_calls",
    }
    for name in ("global_importance.csv", "shap_importance.png", "shap_summary.png"):
        assert (out_dir / name).exists()
    assert len(pd.read_csv(out_dir / "example_reasons.csv")) == 10
