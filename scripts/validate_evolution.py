"""Reproducible local checks and explicitly synthetic examples; no network."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import confusion_matrix

from ml.candidate import CandidateClassifier
from ml.contracts import NewsDocument
from ml.verification import verify

ROOT = Path(__file__).resolve().parents[1]
ASOF = date(2026,10,8)
FACT = "A prefeitura registrou 82 casos em 06/10/2026."


def source(text=FACT, host="a.example", published="2026-10-07", tail="Boletim técnico municipal."):
    return NewsDocument(title="Fixture sintética",body=text+"\n"+tail,
        source_url=f"https://{host}/boletim",
        published_at=datetime.fromisoformat(published).replace(tzinfo=timezone.utc))


def main(candidate, output):
    candidate = candidate.resolve()
    output.mkdir(parents=True,exist_ok=True)
    audit = json.loads((ROOT/"reports/audit_2026-10-08/metrics.json").read_text())
    unchanged = {}
    for name, info in audit["artifacts"].items():
        path = ROOT/name
        unchanged[name] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
    if not all(unchanged.values()): raise RuntimeError("Artefatos de referência divergentes")
    classifier = CandidateClassifier(candidate)
    sources = [source(),source(host="b.example",tail="Relatório independente.")]
    cases = [
        ("corroboração controlada",FACT,sources),
        ("quantidade alterada",FACT.replace("82","8200"),sources),
        ("negação alterada",FACT.replace("registrou","não registrou"),sources),
        ("conflito",FACT,[source(),source(FACT.replace("82","83"),host="b.example")]),
        ("sem evidência",FACT,[]),
        ("parcial",FACT+"\nA universidade registrou 50 casos.",sources),
        ("clickbait","Será que a prefeitura registrou 82 casos?",sources),
        ("sátira","Sátira: a prefeitura registrou 82 casos.",sources),
        ("antiga","A prefeitura registrou 82 casos.",[source("A prefeitura registrou 82 casos.",published="2020-10-07")]),
        ("cópias",FACT,[source(),source(host="b.example")]),
    ]
    examples, latencies = [], []
    for name,text,evidence in cases:
        doc = NewsDocument(body=text)
        start = time.perf_counter()
        result = verify(doc,evidence,ASOF)
        auxiliary = classifier.predict(doc)
        latencies.append((time.perf_counter()-start)*1000)
        examples.append({"case":name,"scope":"synthetic fixtures; not real news or external quality validation",
            "input":text,"result":result.model_dump(mode="json"),"auxiliary":auxiliary})
    import pandas as pd
    predictions = pd.read_csv(candidate/"test_predictions.csv")
    cm = confusion_matrix(predictions.Classe,predictions.predicted,labels=[0,1]).tolist()
    original = json.loads((candidate/"metrics.json").read_text())["confusion_matrix"]
    assert cm == original
    report = {
        "reference_artifacts_unchanged":unchanged,"candidate":str(candidate.relative_to(ROOT)),
        "historical_confusion_matrix_rows_true_cols_prediction":cm,"label_order":["FAKE","REAL"],
        "recent_factual_quality_metrics":None,
        "local_latency_ms":{"p50":float(np.quantile(latencies,.5)),"p95":float(np.quantile(latencies,.95)),"p99":float(np.quantile(latencies,.99)),
            "n":len(latencies),"scope":"synthetic in-process rules + candidate; excludes HTTP, retrieval, DB, concurrency; not SLA"},
        "examples":examples,
    }
    (output/"validation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="examples"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/evolution_2026-10-08")
    args=parser.parse_args()
    main(args.candidate,args.output)
