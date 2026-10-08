import pytest


@pytest.fixture(autouse=True)
def deterministic_optional_models(monkeypatch):
    monkeypatch.setenv("ENABLE_BIAS_MODEL","0")
