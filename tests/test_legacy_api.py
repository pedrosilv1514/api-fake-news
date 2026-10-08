import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.deps import get_db
from api.db.models import NoticiaCache
from api.routers import analise


@pytest.mark.parametrize("expired", [True,False])
def test_legacy_route_with_mock_db_refreshes_old_cache_and_handles_missing_bias(monkeypatch,expired):
    row=NoticiaCache(url="https://example.com/news",titulo="Título antigo",conteudo="texto",
        score_ml=.2,bias_label="indisponível",bias_score=None,
        explicacao_xai=json.dumps({"api_cache_version":"legacy-lexical-v2"}),
        data_processamento=datetime.now(timezone.utc).replace(tzinfo=None)-timedelta(hours=1 if expired else 0))
    session=AsyncMock()
    session.add=Mock()
    result=Mock()
    result.scalars.return_value.first.return_value=row
    session.execute.return_value=result
    async def override():
        yield session
    app.dependency_overrides[get_db]=override
    extracted=Mock(return_value={"title":"Novo título","clean_content":"O governo não registrou 82 casos."})
    monkeypatch.setattr(analise,"extract_news",extracted)
    try:
        with TestClient(app) as client:
            response=client.post("/analisar-url",json={"url":"https://example.com/news"})
        assert response.status_code==200
        data=response.json()
        assert data["confianca_factual"] is None
        assert "score textual" in data["aviso"]
        if expired:
            assert data["titulo"]=="Novo título"
            extracted.assert_called_once()
            session.commit.assert_awaited_once()
        else:
            assert data["titulo"]=="Título antigo"
            extracted.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_unreviewed_feedback_cannot_trigger_reference_mutation():
    from ml.retrain_cron import execute_continuous_training
    with pytest.raises(RuntimeError,match="desativado"):
        execute_continuous_training()

