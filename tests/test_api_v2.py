from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import app
from api.routers import verificacao
from ml.contracts import NewsDocument


def test_v2_runs_without_legacy_model_or_db(monkeypatch):
    monkeypatch.setattr("api.main.engine.load_model",lambda: None)
    with TestClient(app) as client:
        response = client.post("/v2/verificar",json={"document":{"body":"A prefeitura registrou 82 casos."}})
        assert response.status_code==200
        result=response.json()
        assert result["results"][0]["verdict"]=="evidencia_insuficiente"
        assert result["results"][0]["confidence"] is None
        assert result["auxiliary"] == {}


def test_contract_and_ssrf_rejected():
    with TestClient(app) as client:
        assert client.post("/v2/verificar",json={}).status_code==422
        assert client.post("/v2/verificar",json={"document":{"body":"texto"},"url":"https://example.com"}).status_code==422
        assert client.post("/v2/verificar",json={"url":"https://127.0.0.1"}).status_code==400
        assert client.post("/v2/verificar",json={"document":{"body":" "}}).status_code==422
        assert client.post("/v2/verificar",json={"document":{"body":"x"*100001}}).status_code==422
        assert client.post("/v2/verificar",json={"document":{"body":"x"*512001}}).status_code==413


def test_provider_missing_and_source_failure_abstain(monkeypatch):
    from scraper.extractor import ExtractionError
    monkeypatch.delenv("SEARXNG_BASE_URL",raising=False)
    def unavailable(url,**kwargs): raise ExtractionError("offline")
    monkeypatch.setattr(verificacao,"extract_document",unavailable)
    result = verificacao.run_verification(verificacao.VerifyRequest(
        document=NewsDocument(body="A prefeitura registrou 82 casos."),
        search=True,evidence_urls=["https://example.com/source"]))
    assert result.retrieval_status=="search_not_configured"
    assert result.results[0].verdict.value=="evidencia_insuficiente"
    assert any("indisponível" in s for s in result.limitations)


def test_content_and_version_ttl_cache():
    now=[0]
    cache=verificacao.ResultCache(ttl=10,maximum=1,clock=lambda:now[0])
    result=verificacao.run_verification(verificacao.VerifyRequest(document=NewsDocument(body="A prefeitura registrou 82 casos.")))
    cache.put("v1/hash",result)
    assert cache.get("v1/hash") is not None
    assert cache.get("v2/hash") is None
    now[0]=11
    assert cache.get("v1/hash") is None
    cache.put("a",result); cache.put("b",result)
    assert cache.get("a") is None


def test_url_content_changed_is_not_cached(monkeypatch):
    fact="A prefeitura registrou 82 casos."
    state=[fact]
    def fetch(url,**kwargs):
        return NewsDocument(body=state[0],source_url=url,published_at=datetime(2026,10,8,tzinfo=timezone.utc))
    monkeypatch.setattr(verificacao,"extract_document",fetch)
    request=verificacao.VerifyRequest(document=NewsDocument(body=fact),evidence_urls=["https://example.com"])
    first=verificacao.run_verification(request)
    second=verificacao.run_verification(request)
    assert second.cache_hit
    state[0]=fact.replace("82","83")
    third=verificacao.run_verification(request)
    assert not third.cache_hit
    assert first.results[0].evidence[0].relation=="sustenta"
    assert third.results[0].evidence[0].relation=="refuta"
