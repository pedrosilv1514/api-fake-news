"""v2 factual evidence prototype, independent of legacy binary inference/DB."""
import asyncio
from collections import OrderedDict
import hashlib
import json
from functools import partial
import time
import threading
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from pydantic import ConfigDict, HttpUrl, model_validator

from ml.contracts import NewsDocument, VerificationResult
from ml.inference import engine
from api.schemas import AnalyzeUrlRequest, FrontendAnalyzeResponse, VerificationOptions, URL_EXAMPLE
from api.presenters import frontend_response
from ml.verification import SYSTEM_VERSION, extract_claims, verify
from scraper.extractor import ExtractionError, extract_document
from scraper.providers import NoSearchProvider, configured_provider
from scraper.safe_http import FetchError, fetch_public, validate_url

router = APIRouter(prefix="/v2", tags=["Verificação por evidências"])
slots = asyncio.Semaphore(4)


class VerifyRequest(VerificationOptions):
    model_config = ConfigDict(extra="forbid")
    document: NewsDocument | None = None
    url: HttpUrl | None = None

    @model_validator(mode="after")
    def input_contract(self):
        if (self.document is None) == (self.url is None):
            raise ValueError("Forneça document ou url, exclusivamente")
        if self.url and len(str(self.url))>2048:
            raise ValueError("URL muito longa")
        return self


class ResultCache:
    def __init__(self, ttl=300, maximum=128, clock=time.monotonic):
        self.ttl, self.maximum, self.clock = ttl, maximum, clock
        self.items = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            return self._get(key)

    def _get(self, key):
        found = self.items.get(key)
        if not found: return None
        timestamp, result = found
        if self.clock()-timestamp >= self.ttl:
            self.items.pop(key)
            return None
        self.items.move_to_end(key)
        return result.model_copy(deep=True)

    def put(self, key, result):
        with self.lock:
            self._put(key,result)

    def _put(self, key, result):
        self.items[key] = (self.clock(), result.model_copy(deep=True))
        self.items.move_to_end(key)
        while len(self.items)>self.maximum: self.items.popitem(last=False)


cache = ResultCache()


def run_verification(request):
    start = time.perf_counter()
    deadline = time.monotonic()+30
    def download(url):
        remaining = deadline-time.monotonic()
        if remaining<=0: raise ExtractionError("Prazo global de recuperação excedido")
        return extract_document(url,fetcher=partial(fetch_public,timeout=min(8,remaining)))
    urls = [str(u) for u in request.evidence_urls]
    provenance = {u: {"provider": "supplied_url"} for u in urls}
    for u in urls + ([str(request.url)] if request.url else []):
        validate_url(u)
    document = request.document or download(str(request.url))
    status, warnings = "provided_urls", []
    if request.search:
        provider = configured_provider()
        if isinstance(provider, NoSearchProvider):
            status = "search_not_configured"
            warnings.append("Busca automática não configurada; somente URLs fornecidas.")
        else:
            status = "search_provider"
            # Budget: at most two searches and five source downloads.
            for claim in [c for c in extract_claims(document)[0] if c.checkable][:2]:
                if time.monotonic()>=deadline:
                    warnings.append("Prazo global excedido antes da busca.")
                    break
                try:
                    found = provider.search(claim)
                    for k, v in found.items():
                        if k not in provenance: provenance[k] = v
                    urls.extend(found.keys())
                except (FetchError, ValueError):
                    warnings.append("Provedor de busca indisponível.")
    unique = list(dict.fromkeys(urls))[:5]
    sources = []
    for url in unique:
        if time.monotonic()>=deadline:
            warnings.append("Prazo global de recuperação excedido; fontes restantes não consultadas.")
            break
        try: sources.append(download(url))
        except ExtractionError:
            warnings.append(f"Fonte indisponível: {url}")
    key = hashlib.sha256(json.dumps({
        "version":SYSTEM_VERSION, "document":document.content_hash(),
        "sources":[s.content_hash() for s in sources],
        "reference":str(request.reference_date), "status":status, "warnings":warnings,
    },sort_keys=True).encode()).hexdigest()
    # Fetch again before reuse: the cache cannot hide a changed URL's content/date.
    result = cache.get(key)
    if result is not None:
        result.cache_hit = True
        # Provenance must show this access, not the previous download.
        accesses = {str(s.source_url):s.fetched_at for s in sources}
        result.document.fetched_at = document.fetched_at
        for row in result.results:
            for evidence in row.evidence:
                evidence.accessed_at = accesses[str(evidence.url)]
    else:
        result = verify(document,sources,request.reference_date,retrieval_status=status,provenance=provenance)
        result.limitations.extend(warnings)
        cache.put(key,result)
    result.latency_ms = round((time.perf_counter()-start)*1000,3)
    return result


def run_frontend_analysis(request: AnalyzeUrlRequest):
    start = time.perf_counter()
    # Required numeric fields need a real model score, never a fabricated fallback.
    if not engine.ready:
        raise RuntimeError("Classificador textual indisponível")
    result = run_verification(VerifyRequest(**request.model_dump()))
    prediction = engine.predict(result.document.body, str(request.url))
    result.latency_ms = round((time.perf_counter()-start)*1000,3)
    return frontend_response(str(request.url), result, prediction)


async def execute_worker(work):
    try:
        await asyncio.wait_for(slots.acquire(), timeout=1)
    except TimeoutError:
        raise HTTPException(429, "Fila local cheia; tente novamente")
    # Individual downloads have bounded time, size and redirects. Keep the slot until
    # the worker actually finishes (never release while a timed-out thread is running).
    worker = asyncio.create_task(asyncio.to_thread(work))
    def completed(task):
        slots.release()
        if not task.cancelled(): task.exception()  # Retrieve errors even after client disconnect.
    worker.add_done_callback(completed)
    try:
        return await asyncio.shield(worker)
    except (ExtractionError, FetchError) as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@router.post("/analisar-url", response_model=FrontendAnalyzeResponse,
             summary="Analisar URL — saída original + evidências separadas",
             description="Entrada mínima: url. Todos os campos originais ficam na raiz; verificacao contém somente o parecer factual. Busca configurada é tentada por padrão, sem inventar evidências quando indisponível.")
async def analisar_url_frontend(request: AnalyzeUrlRequest):
    if not engine.ready:
        raise HTTPException(503, "Classificador textual indisponível")
    return await execute_worker(lambda: run_frontend_analysis(request))


@router.post("/verificar", response_model=VerificationResult,
             summary="Verificação detalhada por alegação — contrato de auditoria")
async def verificar(request: Annotated[VerifyRequest, Body(openapi_examples={
    "por_url": {"summary":"Extrair a notícia pelo link", "value":{"url":URL_EXAMPLE}},
    "por_texto": {"summary":"Verificar texto fornecido", "value":{"document":{"body":"A prefeitura registrou 82 casos."}}},
})]):
    return await execute_worker(lambda: run_verification(request))
