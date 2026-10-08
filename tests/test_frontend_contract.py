"""Frontend contract regressions; all external extraction/search below is mocked."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from api.deps import get_db
from api.main import app
from api.routers import analise, verificacao
from ml.contracts import NewsDocument

URL = "https://example.com/noticia"
BODY = "A prefeitura registrou 82 casos."
LEGACY_KEYS = {
    "url", "titulo", "prob_fake", "taxa_confiabilidade",
    "bias_label", "bias_score", "janela_analise",
    "o_que_sustenta", "o_que_enfraquece", "o_que_nao_foi_confirmado",
    "modelos_usados",
}
WINDOW_KEYS = {"evidencia", "qualidade_da_fonte", "corroboracao", "contexto", "atualidade"}


@pytest.fixture
def model_and_document(monkeypatch):
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setattr(verificacao.engine, "load_model", lambda: None)
    monkeypatch.setattr(verificacao.engine, "svc_model", object())
    prediction = {
        "prob_fake": .2, "bias_label": "indisponível", "bias_score": None,
        "modelos_usados": ["classificador simulado"],
        "explicacao_xai": {
            "janela_analise": {key: "Explicação controlada." for key in WINDOW_KEYS},
            "o_que_sustenta": ["Correlação textual; não é evidência factual."],
            "o_que_enfraquece": [],
            "o_que_nao_foi_confirmado": ["Fatos sem confirmação independente."],
        },
    }
    predict = Mock(return_value=prediction)
    monkeypatch.setattr(verificacao.engine, "predict", predict)
    doc = NewsDocument(title="Notícia sintética", body=BODY, source_url=URL,
                       published_at=datetime.now(timezone.utc))
    extraction = Mock(return_value=doc)
    monkeypatch.setattr(verificacao, "extract_document", extraction)
    return prediction, doc, extraction, predict


def assert_original_shape(data):
    assert LEGACY_KEYS <= data.keys()
    assert isinstance(data["url"], str)
    assert data["titulo"] is None or isinstance(data["titulo"], str)
    assert isinstance(data["prob_fake"], (int, float))
    assert isinstance(data["taxa_confiabilidade"], (int, float))
    assert data["bias_label"] is None or isinstance(data["bias_label"], str)
    assert data["bias_score"] is None or isinstance(data["bias_score"], (int, float))
    assert set(data["janela_analise"]) == WINDOW_KEYS
    assert all(isinstance(v, str) for v in data["janela_analise"].values())
    for key in ["o_que_sustenta", "o_que_enfraquece", "o_que_nao_foi_confirmado"]:
        assert isinstance(data[key], list)
    assert data["modelos_usados"] is None or isinstance(data["modelos_usados"], list)


def test_url_only_response_retains_all_original_keys(model_and_document):
    prediction, doc, extraction, predict = model_and_document
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url", json={"url": URL})
    assert response.status_code == 200
    data = response.json()
    assert_original_shape(data)
    assert data["titulo"] == doc.title
    assert data["prob_fake"] == prediction["prob_fake"]
    assert data["taxa_confiabilidade"] == 80
    assert data["bias_score"] is None  # never fabricate a tone score
    assert data["confianca_factual"] is None
    assert "score" in data["aviso"]
    assert "document" not in data and "results" not in data
    details = data["verificacao"]
    assert "document" not in details and "auxiliary" not in details
    assert details["results"][0]["verdict"] == "evidencia_insuficiente"
    assert details["results"][0]["confidence"] is None
    assert details["retrieval_status"] == "search_not_configured"
    predict.assert_called_once_with(BODY, URL)
    extraction.assert_called_once()


def test_frontend_v2_matches_v1_on_all_original_fields(model_and_document, monkeypatch):
    prediction, doc, _, _ = model_and_document
    monkeypatch.setattr(analise, "extract_news",
                        lambda url: {"title": doc.title, "clean_content": doc.body})
    session = AsyncMock()
    session.add = Mock()
    query = Mock()
    query.scalars.return_value.first.return_value = None
    session.execute.return_value = query

    async def override():
        yield session
    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            old = client.post("/analisar-url", json={"url": URL})
            new = client.post("/v2/analisar-url", json={"url": URL})
        assert old.status_code == new.status_code == 200
        assert_original_shape(old.json())
        assert_original_shape(new.json())
        for key in LEGACY_KEYS:
            assert old.json()[key] == new.json()[key]
        assert "verificacao" not in old.json()
        session.commit.assert_awaited_once()  # only the original DB-backed route
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("extra", [
    {"document": {"body": "string"}},
    {"reference_date": (date.today()+timedelta(days=1)).isoformat()},
    {"evidence_urls": ["https://example.com"]*6},
])
def test_frontend_input_validation_precedes_extraction(model_and_document, extra):
    _, _, extraction, _ = model_and_document
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url", json={"url": URL, **extra})
    assert response.status_code == 422
    extraction.assert_not_called()


def test_frontend_rejects_private_url(model_and_document):
    _, _, extraction, _ = model_and_document
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url", json={"url": "https://127.0.0.1"})
    assert response.status_code == 400
    extraction.assert_not_called()


def test_missing_text_model_is_503_not_fabricated_numeric_scores(model_and_document, monkeypatch):
    _, _, extraction, predict = model_and_document
    monkeypatch.setattr(verificacao.engine, "svc_model", None)
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url", json={"url": URL})
    assert response.status_code == 503
    assert "prob_fake" not in response.json()
    extraction.assert_not_called()
    predict.assert_not_called()


def test_configured_search_is_attempted_without_frontend_supplying_sources(model_and_document, monkeypatch):
    provider = Mock()
    provider.search.return_value = []
    monkeypatch.setattr(verificacao, "configured_provider", lambda: provider)
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url", json={"url": URL})
    assert response.status_code == 200
    provider.search.assert_called_once()
    assert response.json()["verificacao"]["retrieval_status"] == "search_provider"


def test_search_can_be_disabled_explicitly(model_and_document, monkeypatch):
    provider_factory = Mock()
    monkeypatch.setattr(verificacao, "configured_provider", provider_factory)
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url", json={"url": URL, "search": False})
    assert response.status_code == 200
    provider_factory.assert_not_called()


def test_detailed_audit_route_keeps_its_existing_output(model_and_document):
    with TestClient(app) as client:
        response = client.post("/v2/verificar", json={"url": URL})
    assert response.status_code == 200
    assert {"document", "results", "summary", "system_version"} <= response.json().keys()
    assert "prob_fake" not in response.json()


def test_openapi_has_original_required_fields_and_url_only_examples():
    schema = app.openapi()
    assert not schema["paths"]["/analisar-url"]["post"].get("deprecated", False)
    for name in ["AnalyzeResponse", "FrontendAnalyzeResponse"]:
        required = schema["components"]["schemas"][name]["required"]
        assert LEGACY_KEYS <= set(required)
    assert "verificacao" in schema["components"]["schemas"]["FrontendAnalyzeResponse"]["required"]
    for name in ["AnalyzeRequest", "AnalyzeUrlRequest"]:
        example = schema["components"]["schemas"][name]["examples"][0]
        assert set(example) == {"url"}
        assert "example.com" not in example["url"]
    examples = schema["paths"]["/v2/verificar"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert set(examples["por_url"]["value"]) == {"url"}
    assert set(examples["por_texto"]["value"]) == {"document"}


def test_new_url_endpoint_with_real_reference_model(monkeypatch):
    # Real persisted classifier, deterministic mock HTML extraction: no network/DB.
    monkeypatch.setattr(verificacao, "extract_document",
                        lambda url, **kwargs: NewsDocument(title="Fixture", body=BODY, source_url=URL))
    with TestClient(app) as client:
        expected = verificacao.engine.predict(BODY, URL)
        response = client.post("/v2/analisar-url", json={"url": URL, "search": False})
    assert response.status_code == 200
    assert response.json()["prob_fake"] == expected["prob_fake"]
    assert response.json()["taxa_confiabilidade"] == round((1-expected["prob_fake"])*100, 2)
    assert response.json()["confianca_factual"] is None


@pytest.mark.parametrize("counts,verdict", [
    (["83", "83"], "refutada"),
    (["82", "83"], "evidencias_conflitantes"),
])
def test_factual_conclusion_never_overwrites_original_scores(model_and_document, monkeypatch, counts, verdict):
    prediction, document, _, _ = model_and_document
    urls = ["https://a.example/boletim", "https://b.example/boletim"]
    sources = {
        url: NewsDocument(title="Fonte sintética", source_url=url,
            body=f"A prefeitura registrou {count} casos.\nRelatório da equipe {i}.",
            published_at=datetime.now(timezone.utc))
        for i, (url, count) in enumerate(zip(urls, counts))
    }

    def extract(url, **kwargs):
        return document if url == URL else sources[url]
    monkeypatch.setattr(verificacao, "extract_document", extract)
    with TestClient(app) as client:
        response = client.post("/v2/analisar-url",
            json={"url": URL, "evidence_urls": urls, "search": False})
    assert response.status_code == 200
    data = response.json()
    assert_original_shape(data)
    assert data["prob_fake"] == prediction["prob_fake"]
    assert data["taxa_confiabilidade"] == 80
    row = next(r for r in data["verificacao"]["results"] if r["claim"]["text"] == BODY)
    assert row["verdict"] == verdict
    assert row["confidence"] is None
    assert {e["url"] for e in row["evidence"]} == set(urls)

