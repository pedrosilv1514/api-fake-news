from fastapi.testclient import TestClient

from api.main import app
from ml.inference import engine


def test_health_and_readiness_enter_lifespan():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["ml_ready"] is True
        assert response.json()["bias_ready"] is False
        assert response.json()["factual_validation"] == "pending"
        assert client.get("/ready").status_code == 200


def test_liveness_is_not_readiness(monkeypatch):
    def missing():
        engine.svc_model = None
        engine.hf_classifier = None
    monkeypatch.setattr(engine,"load_model",missing)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["ml_ready"] is False
        assert client.get("/ready").status_code == 503
