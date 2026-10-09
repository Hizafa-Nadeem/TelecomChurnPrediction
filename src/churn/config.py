"""Parameter models (configs/params.yaml) and environment settings (.env)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    openml_id: int
    target: str
    id_column: str


class SplitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(gt=0, lt=1)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(ge=1)


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catboost_iterations: int = Field(ge=1)
    logreg_c_grid: list[float]


class TuneParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(ge=1)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(ge=1)


class BusinessParams(BaseModel):
    """Assumptions for the retention value; the dataset has no customer-value column."""

    model_config = ConfigDict(extra="forbid")

    remaining_months: int = Field(ge=1)
    acceptance_rate: float = Field(gt=0, le=1)
    offer_cost: float = Field(gt=0)


class Params(BaseModel):
    """Contents of configs/params.yaml."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    data: DataParams
    split: SplitParams
    train: TrainParams
    tune: TuneParams
    business: BusinessParams


class Settings(BaseSettings):
    """Environment-specific settings and secrets, read from env vars or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "telecom-churn"
    mlflow_model_name: str = "churn-catboost"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


def load_params(path: str | Path) -> Params:
    """Load and validate the parameter file."""
    with open(path, encoding="utf-8") as f:
        return Params.model_validate(yaml.safe_load(f))
