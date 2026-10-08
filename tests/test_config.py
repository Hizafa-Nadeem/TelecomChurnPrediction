from pathlib import Path

import pytest
from pydantic import ValidationError

from churn.config import Params, Settings, load_params

PARAMS_PATH = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"


def test_params_file_is_valid() -> None:
    params = load_params(PARAMS_PATH)
    assert params.data.openml_id == 40701
    assert params.split.test_size == 0.2


def test_unknown_param_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Params.model_validate({"seed": 1, "typo": 2})


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9000")
    assert Settings().api_port == 9000
