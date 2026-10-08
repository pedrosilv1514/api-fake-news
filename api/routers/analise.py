import asyncio
import json
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.presenters import dict_to_response
from api.deps import get_db
from api.db.models import NoticiaCache
from scraper.extractor import extract_news, ExtractionError
from ml.inference import engine
from scraper.safe_http import FetchError, validate_url

router = APIRouter(tags=["Análise"])

@router.post("/analisar-url", response_model=AnalyzeResponse,
             summary="Analisar notícia por URL — contrato original do frontend",
             description="Recebe apenas a URL e mantém os campos originais. Scores são textuais; para evidências no mesmo formato, use /v2/analisar-url.")
async def analisar_url(request: AnalyzeRequest, session: AsyncSession = Depends(get_db)):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Classificador textual indisponível")
    url_str = str(request.url)
    try:
        validate_url(url_str)
    except FetchError as exc:
        raise HTTPException(400, str(exc))
    
    # 1. Verificar cache no BD
    query = select(NoticiaCache).where(NoticiaCache.url == url_str)
    result = await session.execute(query)
    noticia_db = result.scalars().first()
    
    if noticia_db and noticia_db.data_processamento and noticia_db.data_processamento > datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5):
        prob = noticia_db.score_ml or 0.0
        try:
            xai_dict = json.loads(noticia_db.explicacao_xai) if noticia_db.explicacao_xai else {}
        except:
            xai_dict = {}
        if xai_dict.get("api_cache_version") == "legacy-lexical-v2":
            return dict_to_response(noticia_db.url, noticia_db.titulo, prob, noticia_db.bias_label, noticia_db.bias_score, xai_dict, ["Cache DB (textual; TTL 5 min)"])

    # 2. Extração de texto usando Threads para não bloquear o loop assíncrono
    try:
        extracted_data = await asyncio.to_thread(extract_news, url_str)
    except ExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de extração: {e}")

    # 3. Inferência de ML em Thread para não bloquear
    try:
        prediction = await asyncio.to_thread(engine.predict, extracted_data["clean_content"], url_str)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência ML: {e}")

    xai_dict = prediction.get("explicacao_xai") or {}
    xai_dict["api_cache_version"] = "legacy-lexical-v2"
    
    # 4. Salvar no banco
    nova_noticia = noticia_db or NoticiaCache(url=url_str)
    nova_noticia.titulo = extracted_data["title"]
    nova_noticia.conteudo = extracted_data["clean_content"]
    nova_noticia.score_ml = prediction["prob_fake"]
    nova_noticia.bias_label = prediction["bias_label"]
    nova_noticia.bias_score = prediction["bias_score"]
    nova_noticia.explicacao_xai = json.dumps(xai_dict, ensure_ascii=False)
    nova_noticia.data_processamento = datetime.now(timezone.utc).replace(tzinfo=None)
    
    session.add(nova_noticia)
    await session.commit()
    await session.refresh(nova_noticia)

    # 5. Retornar resposta
    prob = prediction["prob_fake"]
    return dict_to_response(
        url=nova_noticia.url,
        titulo=nova_noticia.titulo,
        prob_fake=prob,
        bias_label=prediction["bias_label"],
        bias_score=prediction["bias_score"],
        xai_dict=xai_dict,
        modelos_usados=prediction["modelos_usados"]
    )

from api.schemas import FeedbackRequest
from api.db.models import VotoComunidade

@router.post("/votar")
async def votar(request: FeedbackRequest, session: AsyncSession = Depends(get_db)):
    url_str = str(request.url)
    
    # Busca a notícia no cache para vincular o voto
    query = select(NoticiaCache).where(NoticiaCache.url == url_str)
    result = await session.execute(query)
    noticia_db = result.scalars().first()
    
    if not noticia_db:
        raise HTTPException(status_code=404, detail="Notícia não encontrada no sistema para votar.")
        
    # Registrar o voto
    novo_voto = VotoComunidade(
        id_noticia=noticia_db.id,
        id_usuario=request.id_usuario,
        voto=request.voto
    )
    
    session.add(novo_voto)
    await session.commit()
    
    return {"status": "success", "message": "Feedback recebido, sujeito a revisão; não é rótulo de treinamento."}
