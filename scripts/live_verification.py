"""Opt-in public HTTPS validation. Never substitutes for blinded human annotation."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time
from fastapi.testclient import TestClient
from api.main import app

from ml.contracts import NewsDocument
from ml.verification import sentences, verify
from scraper.extractor import ExtractionError, extract_document

URLS = [
    "https://www.santos.sp.gov.br/?q=noticia/santos-define-orgaos-que-atuarao-no-plano-preventivo-da-defesa-civil",
    "https://www.santos.sp.gov.br/?q=noticia/ilha-diana-em-santos-recebe-equipamentos-para-atendimento-em-situacoes-de-emergencia",
]


def main(output, reference):
    rows=[]
    for url in URLS:
        try:
            started=time.perf_counter()
            source=extract_document(url)
            retrieval_ms=(time.perf_counter()-started)*1000
            excerpt=next((s for s in sentences(source.body) if len(s.split())>=5 and len(s)<500
                          and s != source.title and not s.startswith(("Publicado", "Atualizado"))), source.body[:400])
            # This is a statement taken from the source; its publication isn't a truth label.
            doc=NewsDocument(body=excerpt, published_at=source.published_at)
            result=verify(doc,[source],reference,retrieval_status="real_https_public_source")
            with TestClient(app) as client:
                started=time.perf_counter()
                response=client.post("/v2/verificar",json={"document":doc.model_dump(mode="json"),
                    "evidence_urls":[url],"reference_date":str(reference)})
                api_ms=(time.perf_counter()-started)*1000
                response.raise_for_status()
            rows.append({
                "url":str(source.source_url),"title":source.title,
                "published_at":source.published_at.isoformat() if source.published_at else None,
                "updated_at":source.updated_at.isoformat() if source.updated_at else None,
                "date_provenance":source.date_provenance,
                "accessed_at":source.fetched_at.isoformat(),"content_hash":source.content_hash(),
                "recent_within_7_days":bool(source.published_at and 0 <= (reference-source.published_at.date()).days<=7),
                "retrieval_and_extraction_ms":round(retrieval_ms,3),
                "api_status_code":response.status_code,"api_local_latency_ms":round(api_ms,3),
                "api_result":response.json(),
                "result":result.model_dump(mode="json"),
                "scope":"real retrieved source; one origin only; no independent human truth annotation",
            })
        except ExtractionError as exc:
            chain, cause = [], exc
            while cause is not None:
                chain.append(type(cause).__name__)
                cause = cause.__cause__
            rows.append({"url":url,"error":str(exc),"failure_types":chain,"scope":"failed real network retrieval"})
    output.mkdir(parents=True,exist_ok=True)
    report={"reference_date":str(reference),"items":rows,"blind_evaluation":False,"quality_metrics":None}
    (output/"live_retrieval.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if all("error" not in row for row in rows) else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("reports/evolution_2026-10-08"))
    parser.add_argument("--reference",type=date.fromisoformat,default=date(2026,10,8))
    args=parser.parse_args()
    sys.exit(main(args.output,args.reference))
