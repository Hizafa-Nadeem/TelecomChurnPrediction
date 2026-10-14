from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
from api.main import app
from fastapi.testclient import TestClient

from churn.package import package
from churn.pyfunc_model import ALIAS, predictor_columns
from churn.split import TRAIN_FILE
from helpers import MODEL_NAME, PARAMS, TARGET, registry_settings


@pytest.fixture
def client(workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The API running against a freshly packaged and registered tiny model."""
    package(
        PARAMS,
        workspace["processed"],
        workspace["models"],
        workspace["calibration"],
        workspace["evaluation"],
        registry_settings(),
    )
    monkeypatch.setenv("MODEL_URI", f"models:/{MODEL_NAME}@{ALIAS}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def customers(workspace: dict[str, Path]) -> list[dict[str, Any]]:
    frame = pd.read_csv(workspace["processed"] / TRAIN_FILE, dtype={"state": str, "area_code": str})
    rows = frame[predictor_columns(TARGET)].head(6).to_dict(orient="records")
    return [{str(k): v for k, v in row.items()} for row in rows]


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["threshold"] == 0.3
    assert len(info["input_columns"]) == 19
    assert info["model_uri"].endswith(f"@{ALIAS}")


def test_predict_returns_probability_decision_and_reasons(
    client: TestClient, customers: list[dict[str, Any]]
) -> None:
    body = client.post("/predict", json=customers[0]).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["decision"] == int(body["churn_probability"] >= 0.3)
    assert isinstance(body["reasons"], list)
    assert len(body["reasons"]) <= 3
    assert all(isinstance(r, str) and "=" in r for r in body["reasons"])


def test_batch_matches_single_predictions_in_order(
    client: TestClient, customers: list[dict[str, Any]]
) -> None:
    batch = client.post("/predict/batch", json={"customers": customers}).json()["predictions"]
    singles = [client.post("/predict", json=c).json() for c in customers]
    assert len(batch) == len(customers)
    assert batch == singles


def test_numeric_state_and_area_code_are_accepted(
    client: TestClient, customers: list[dict[str, Any]]
) -> None:
    payload = {**customers[0], "state": int(customers[0]["state"]), "area_code": 415}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("number_customer_service_calls", 10),
        ("area_code", "999"),
        ("state", "51"),
        ("total_day_minutes", -1.0),
        ("international_plan", 2),
        ("account_length", 0),
        ("phone_number", 12345),
        ("class", 1),
    ],
)
def test_invalid_input_is_rejected(
    client: TestClient, customers: list[dict[str, Any]], field: str, value: object
) -> None:
    response = client.post("/predict", json={**customers[0], field: value})
    assert response.status_code == 422


def test_missing_field_and_bad_batch_sizes_are_rejected(
    client: TestClient, customers: list[dict[str, Any]]
) -> None:
    incomplete = {k: v for k, v in customers[0].items() if k != "account_length"}
    assert client.post("/predict", json=incomplete).status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    too_many = {"customers": [customers[0]] * 1001}
    assert client.post("/predict/batch", json=too_many).status_code == 422
