"""Request and response models; they mirror docs/data_contract.md so bad input is rejected."""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

StrCode = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, int) else v)]
Flag = Literal[0, 1]


class Customer(BaseModel):
    """One customer: the 19 predictors of the data contract (no phone_number, no class)."""

    model_config = ConfigDict(extra="forbid")

    state: StrCode = Field(description="Anonymous state code, 0-50")
    area_code: StrCode = Field(description="408, 415 or 510")
    account_length: int = Field(ge=1)
    international_plan: Flag
    voice_mail_plan: Flag
    number_vmail_messages: int = Field(ge=0)
    number_customer_service_calls: int = Field(ge=0, le=9)
    total_day_minutes: float = Field(ge=0)
    total_day_calls: int = Field(ge=0)
    total_day_charge: float = Field(ge=0)
    total_eve_minutes: float = Field(ge=0)
    total_eve_calls: int = Field(ge=0)
    total_eve_charge: float = Field(ge=0)
    total_night_minutes: float = Field(ge=0)
    total_night_calls: int = Field(ge=0)
    total_night_charge: float = Field(ge=0)
    total_intl_minutes: float = Field(ge=0)
    total_intl_calls: int = Field(ge=0)
    total_intl_charge: float = Field(ge=0)

    @field_validator("state")
    @classmethod
    def _state_in_range(cls, value: str) -> str:
        if value not in {str(i) for i in range(51)}:
            raise ValueError("state must be a code from 0 to 50")
        return value

    @field_validator("area_code")
    @classmethod
    def _area_code_known(cls, value: str) -> str:
        if value not in {"408", "415", "510"}:
            raise ValueError("area_code must be 408, 415 or 510")
        return value


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    decision: int = Field(ge=0, le=1, description="1 = make a retention offer")
    reasons: list[str] = Field(description="Up to 3 features raising this customer's risk")


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_uri: str
    run_id: str | None
    threshold: float
    input_columns: list[str]
