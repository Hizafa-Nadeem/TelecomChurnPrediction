"""FastAPI service for the registered churn model.

    uvicorn api.main:app --host 127.0.0.1 --port 8000

Settings come from environment variables: `MODEL_URI` (default: the `production` alias of the
registered model, or a local exported model folder in Docker) and `MLFLOW_TRACKING_URI`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings

from api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    ModelInfo,
    Prediction,
)

REASON_COLUMNS = ["reason_1", "reason_2", "reason_3"]


class ApiSettings(BaseSettings):
    model_uri: str = "models:/churn-catboost@production"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at start-up."""
    settings = ApiSettings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    app.state.model_uri = settings.model_uri
    app.state.model = mlflow.pyfunc.load_model(settings.model_uri)
    yield


app = FastAPI(title="Telecom churn prediction", version="1.0", lifespan=lifespan)


def predict_frame(model: Any, customers: list[Customer]) -> list[Prediction]:
    """Score customers with the model and shape the output table into responses."""
    frame = pd.DataFrame([c.model_dump() for c in customers])
    out = model.predict(frame)
    return [
        Prediction(
            churn_probability=float(row.churn_probability),
            decision=int(row.decision),
            reasons=[r for r in (getattr(row, c) for c in REASON_COLUMNS) if r],
        )
        for row in out.itertuples(index=False)
    ]


@app.get("/health")
def health() -> dict[str, str]:
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    model = app.state.model
    inner = model.unwrap_python_model()
    return ModelInfo(
        model_uri=app.state.model_uri,
        run_id=model.metadata.run_id,
        threshold=inner.threshold,
        input_columns=inner.input_columns,
    )


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return predict_frame(app.state.model, [customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=predict_frame(app.state.model, request.customers))
