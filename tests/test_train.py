import numpy as np
import pandas as pd
import pytest

from churn.train import build_models, evaluate_model
from helpers import PARAMS, TARGET, signal_frame

FAST = PARAMS.model_copy(
    update={
        "train": PARAMS.train.model_copy(update={"catboost_iterations": 20}),
        "split": PARAMS.split.model_copy(update={"cv_folds": 3, "cv_repeats": 1}),
    }
)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    df = signal_frame()
    return df.drop(columns=TARGET), df[TARGET]


def test_all_variants_are_built() -> None:
    assert list(build_models(FAST)) == [
        "dummy",
        "logreg",
        "catboost",
        "catboost_weighted",
        "catboost_smotenc",
    ]


def test_every_variant_scores_in_range_and_beats_the_dummy(
    data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = data
    results = {
        name: evaluate_model(model, X, y, FAST, name) for name, model in build_models(FAST).items()
    }
    assert results["dummy"]["cv_pr_auc_mean"] == pytest.approx(y.mean(), abs=0.02)
    for name, metrics in results.items():
        assert 0 <= metrics["cv_pr_auc_mean"] <= 1
        assert 0 <= metrics["cv_roc_auc_mean"] <= 1
        if name != "dummy":
            assert metrics["cv_pr_auc_mean"] > results["dummy"]["cv_pr_auc_mean"] + 0.1


def test_same_seed_gives_identical_metrics(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    models = build_models(FAST)
    first = evaluate_model(models["catboost"], X, y, FAST, "catboost")
    second = evaluate_model(models["catboost"], X, y, FAST, "catboost")
    assert first == second


def test_more_service_calls_or_intl_plan_never_lowers_logreg_risk(
    data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = data
    model = build_models(FAST)["logreg"].fit(X, y)
    base = X.iloc[[0]].copy()
    base["number_customer_service_calls"] = 0
    base["international_plan"] = 0
    risk = []
    for calls in range(7):
        row = base.copy()
        row["number_customer_service_calls"] = calls
        risk.append(model.predict_proba(row)[0, 1])
    with_plan = base.copy()
    with_plan["international_plan"] = 1
    assert np.all(np.diff(risk) >= -1e-9)
    assert model.predict_proba(with_plan)[0, 1] >= risk[0]
