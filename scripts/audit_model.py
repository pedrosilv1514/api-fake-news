"""Reproduce the 2026-10-08 diagnostic without training or changing serving artifacts.

Run from the repository root with the existing virtual environment:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python scripts/audit_model.py
Only reports/audit_2026-10-08/ is written. External sources are not fetched here.
"""

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sqlite3
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)

from ml.preprocessing import limpar_texto

OUT = ROOT / "reports/audit_2026-10-08"
SOURCES = {
    "mte": "https://agenciagov.ebc.com.br/noticias/202610/201clista-suja201d-do-trabalho-analogo-a-escravidao-e-atualizada-com-82-novos-empregadores",
    "faltou": "https://www.tse.jus.br/comunicacao/radio/2026/Outubro/fato-ou-boato-mensagem-faltou-nao-significa-que-eleitor-deixou-de-votar",
    "resultados": "https://www.tse.jus.br/comunicacao/radio/2026/Outubro/fato-ou-boato-justica-eleitoral-totaliza-e-divulga-os-resultados-no-mesmo-dia-da-eleicao",
    "historica": "https://noticias.uol.com.br/internacional/ultimas-noticias/2020/03/27/equador-prepara-cova-coletiva-para-mortos-por-covid-19-em-guaiaquil.htm",
}

# Labels and wording fixed before running predictions. These are authored
# summaries/controlled perturbations, not a representative blind news corpus.
EMPLOYERS = "Em 6 de outubro de 2026, o Ministério do Trabalho informou a inclusão de 82 empregadores no cadastro de trabalho análogo à escravidão. Os casos envolvem 553 trabalhadores e foram identificados entre 2024 e 2026."
ABSENCE = 'Segundo esclarecimento do TSE de 7 de outubro de 2026, a mensagem "faltou" no serviço de justificativa não confirma ausência na votação. Os registros estão em processamento e a atualização é prevista até 11 de outubro.'
CASES = [
    {"id": "R1", "kind": "resumo recente", "expected": 1, "source": SOURCES["mte"], "source_date": "2026-10-06", "text": EMPLOYERS},
    {"id": "R2", "kind": "alteração numérica sintética", "expected": 0, "source": SOURCES["mte"], "source_date": "2026-10-06", "text": EMPLOYERS.replace("82 empregadores", "8200 empregadores")},
    {"id": "R3", "kind": "resumo recente com negação", "expected": 1, "source": SOURCES["faltou"], "source_date": "2026-10-07", "text": ABSENCE},
    {"id": "R4", "kind": "negação removida, sintético", "expected": 0, "source": SOURCES["faltou"], "source_date": "2026-10-07", "text": ABSENCE.replace("não confirma", "confirma")},
    {"id": "R5", "kind": "resumo de esclarecimento recente", "expected": 1, "source": SOURCES["resultados"], "source_date": "2026-10-06", "text": "O TSE esclareceu em 6 de outubro de 2026 que a interrupção no aplicativo Resultados, durante a eleição de 4 de outubro, atingiu a consulta pública. A contagem dos votos continuou sem alteração."},
    {"id": "R6", "kind": "paráfrase de alegação desmentida", "expected": 0, "source": SOURCES["resultados"], "source_date": "2026-10-06", "text": "A interrupção no aplicativo Resultados em 4 de outubro de 2026 modificou a contagem dos votos e alterou o resultado da eleição presidencial. A falha na consulta pública mudou a apuração oficial."},
    {"id": "E7", "kind": "fragmento insuficiente", "expected": None, "source": SOURCES["mte"], "source_date": "2026-10-06", "text": "Ministério atualiza cadastro. Novos nomes foram incluídos."},
    {"id": "E8", "kind": "clickbait com fato preservado", "expected": 1, "source": SOURCES["mte"], "source_date": "2026-10-06", "text": "VOCÊ NÃO VAI ACREDITAR! Veja o que aconteceu com 82 empregadores! Em 6 de outubro de 2026, o Ministério do Trabalho incluiu esses empregadores no cadastro de trabalho análogo à escravidão."},
    {"id": "E9", "kind": "sarcasmo sintético", "expected": None, "source": None, "source_date": None, "text": "Claro, agora as urnas têm uma bola de cristal e adivinham o voto antes de o eleitor chegar. Que maravilha de tecnologia! Esta frase é uma ironia, não uma reportagem."},
    {"id": "E10", "kind": "notícia histórica, rótulo do corpus", "expected": 1, "source": SOURCES["historica"], "source_date": "2020-03-27", "text": "Em 27 de março de 2020, o governo do Equador preparava uma cova coletiva em Guaiaquil para vítimas da covid-19. A informação descreve uma situação daquele período, e não um acontecimento atual."},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def latency_summary(samples):
    return {"n": len(samples), "unit": "ms", "p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95)), "min": float(min(samples)), "max": float(max(samples))}


def metrics(y, pred, prob_fake=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tp, fn, fp, tn = (int(x) for x in cm.ravel())
    result = {
        "n": len(y), "label_order": ["FAKE (0)", "REAL (1)"],
        "confusion_matrix_rows_actual_cols_predicted": cm.tolist(),
        "accuracy": accuracy_score(y, pred),
        "precision_fake": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall_fake": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1_fake": f1_score(y, pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_weighted": f1_score(y, pred, average="weighted"),
        "false_positive_rate_true_marked_fake": fp / (fp + tn),
        "false_negative_rate_fake_marked_true": fn / (fn + tp),
        "counts_positive_means_fake": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }
    if prob_fake is not None:
        truth_fake = (y == 0).astype(int)
        prob_fake = np.asarray(prob_fake)
        result.update({"brier_loss_fake": brier_score_loss(truth_fake, prob_fake), "log_loss": log_loss(truth_fake, prob_fake), "roc_auc_fake": roc_auc_score(truth_fake, prob_fake)})
        bins = []
        for i in range(10):
            mask = (prob_fake >= i / 10) & (prob_fake < (i + 1) / 10 if i < 9 else prob_fake <= 1)
            if mask.any():
                bins.append({"lower": i / 10, "upper": (i + 1) / 10, "n": int(mask.sum()), "mean_probability": float(prob_fake[mask].mean()), "observed_fake_fraction": float(truth_fake[mask].mean())})
        result["calibration_bins_in_this_nonindependent_sample"] = bins
        result["ece_10_bins_nonindependent"] = sum(b["n"] / len(y) * abs(b["mean_probability"] - b["observed_fake_fraction"]) for b in bins)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = joblib.load(ROOT / "models/best_linearsvc_model.joblib")
    calibrated = joblib.load(ROOT / "models/calibrated_pipeline.joblib")
    root_data = pd.read_csv(ROOT / "fake_recogna_limpo.csv")
    full = pd.read_csv(ROOT / "data/fake_recogna_limpo.csv")
    train = pd.read_csv(ROOT / "notebooks/v2_pipeline_linearsvc/data/processed/train.csv")
    holdout = pd.read_csv(ROOT / "notebooks/v2_pipeline_linearsvc/data/processed/test_holdout.csv")
    title_clean = root_data.Titulo.fillna("").map(limpar_texto)
    cc = calibrated.calibrated_classifiers_[0]
    scaler = cc.calibrators[0]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic; no fit, retraining, artifact replacement, external writes, or production DB writes",
        "environment": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(), "versions": {n: importlib.metadata.version(n) for n in ["scikit-learn", "numpy", "pandas", "joblib", "torch", "transformers", "nltk", "fastapi"]}},
        "artifacts": {str(p): {"sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p in ["models/best_linearsvc_model.joblib", "models/calibrated_pipeline.joblib", "fake_recogna_limpo.csv", "data/fake_recogna_limpo.csv", "notebooks/v2_pipeline_linearsvc/data/processed/train.csv", "notebooks/v2_pipeline_linearsvc/data/processed/test_holdout.csv"]},
        "provenance": {
            "corpus_rows": len(full), "corpus_class_counts": {str(k): int(v) for k,v in full.Classe.value_counts().items()},
            "root_and_data_clean_text_equal": root_data.texto_limpo.equals(full.texto_limpo),
            "full_text_exact_duplicates": int(full.texto_limpo.duplicated().sum()),
            "train_rows": len(train), "holdout_rows": len(holdout),
            "train_class_counts": {str(k): int(v) for k,v in train.label.value_counts().items()},
            "holdout_class_counts": {str(k): int(v) for k,v in holdout.label.value_counts().items()},
            "exact_train_holdout_text_overlap": len(set(train.texto_limpo) & set(holdout.texto_limpo)),
            "holdout_titles_present_in_rows_of_calibration_corpus": int(holdout.texto_limpo.isin(title_clean).sum()),
            "train_titles_present_in_rows_of_calibration_corpus": int(train.texto_limpo.isin(title_clean).sum()),
            "vocabulary_size": len(base["tfidf"].vocabulary_),
            "train_clean_word_count_mean": float(train.texto_limpo.str.split().str.len().mean()),
            "full_clean_word_count_mean": float(full.texto_limpo.fillna("").str.split().str.len().mean()),
            "raw_base_encoding_from_notebook": {"0": "REAL, original label 1", "1": "FAKE, original label 0"},
            "calibrated_output_encoding_from_calibration_script": {"0": "FAKE", "1": "REAL"},
            "calibrator_sigmoid_a": float(scaler.a_), "calibrator_sigmoid_b": float(scaler.b_),
            "base_and_calibrator_frozen_weights_equal": bool(np.array_equal(base["clf"].coef_, cc.estimator.estimator["clf"].coef_)),
            "important_limitation": "Holdout articles were in calibration data; full corpus is calibration fit data. Neither is an independent production evaluation. Base holdout tests titles only.",
        },
        "evaluations": {},
    }
    for name, df, label_col in [("title_holdout", holdout, "label"), ("full_calibration_corpus", full, "Classe")]:
        texts = df.texto_limpo.fillna("").astype(str).tolist()
        clean = list(map(limpar_texto, texts))
        prob = calibrated.predict_proba(clean)[:, list(calibrated.classes_).index(0)]
        pred = np.where(prob >= .5, 0, 1)
        report["evaluations"][name + "_serving"] = metrics(df[label_col], pred, prob)
        report["evaluations"][name + "_base_corrected_label_interpretation"] = metrics(df[label_col], 1 - base.predict(texts))
        pd.DataFrame({"row_index": df.index, "text": texts, "actual_label": df[label_col].astype(int), "predicted_label": pred, "prob_fake_unrounded": prob, "prob_fake_api_rounded": np.round(prob, 4), "correct": df[label_col].to_numpy() == pred}).to_csv(OUT / (name + "_predictions.csv"), index=False)
    report["base_training_accuracy_titles_correct_label_mapping"] = float(accuracy_score(train.label.astype(int), 1-base.predict(train.texto_limpo)))
    cm = report["evaluations"]["title_holdout_serving"]["confusion_matrix_rows_actual_cols_predicted"]
    pd.DataFrame(cm, index=["actual_FAKE", "actual_REAL"], columns=["predicted_FAKE", "predicted_REAL"]).to_csv(OUT / "confusion_matrix.csv")
    # Export an SVG matrix that can be used in a presentation without extra packages.
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="650" height="390" viewBox="0 0 650 390"><rect width="650" height="390" fill="white"/><g font-family="sans-serif" fill="#17213a"><text x="25" y="35" font-size="21">Serving artifact: 2,305 title holdout cases</text><text x="25" y="62" font-size="13">Not independent: articles were included in calibration</text><text x="290" y="100">Predicted FAKE</text><text x="450" y="100">Predicted REAL</text><text x="25" y="167">Actual FAKE</text><text x="25" y="267">Actual REAL</text>'
    for row in range(2):
        for col in range(2):
            x, y = 245 + 175 * col, 115 + 100 * row
            fill = '#d1fae5' if row == col else '#fee2e2'
            svg += f'<rect x="{x}" y="{y}" width="160" height="85" rx="6" fill="{fill}"/><text x="{x+80}" y="{y+52}" text-anchor="middle" font-size="30">{cm[row][col]}</text>'
    svg += '<text x="25" y="350" font-size="14">Positive class = FAKE. Accuracy = 90.33%. Threshold = 0.50.</text></g></svg>\n'
    (OUT / "confusion_matrix.svg").write_text(svg)
    # Same NLPEngine used by FastAPI, with both real models, loaded offline.
    from ml.inference import NLPEngine
    e = NLPEngine()
    started = time.perf_counter()
    e.load_model()
    report["benchmark"] = {"model_load_seconds_cached_excludes_module_imports": time.perf_counter()-started, "hf_device": str(e.hf_classifier.device), "hf_parameter_count": sum(p.numel() for p in e.hf_classifier.model.parameters()), "hf_weight_bytes": sum(p.numel()*p.element_size() for p in e.hf_classifier.model.parameters())}
    example_results = []
    for case in CASES:
        started = time.perf_counter()
        result = e.predict(case["text"], case["source"] or "")
        elapsed = (time.perf_counter()-started)*1000
        p = result["prob_fake"]
        pred = 0 if p >= .5 else 1
        item = {**case, "clean_text": limpar_texto(case["text"]), "predicted_label": pred, "predicted_class": "FAKE" if pred == 0 else "REAL", "confidence_predicted_class": max(p, 1-p), "api_taxa_confiabilidade": round((1-p)*100,2), "correct_when_binary_ground_truth": pred == case["expected"] if case["expected"] is not None else None, "latency_ms": elapsed, **result}
        example_results.append(item)
        print(case["id"], item["predicted_class"], "prob_fake",p,"confidence",round(item["confidence_predicted_class"],4),"bias",result["bias_label"],result["bias_score"],flush=True)
    save_json(OUT / "examples.json", example_results)
    fields = ["id", "kind", "source_date", "source", "expected", "predicted_class", "prob_fake", "confidence_predicted_class", "api_taxa_confiabilidade", "bias_label", "bias_score", "latency_ms", "correct_when_binary_ground_truth", "text", "clean_text"]
    with (OUT / "examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(example_results)
    report["edge_pairs"] = {"number_R1_R2_identical_linear_input": example_results[0]["clean_text"] == example_results[1]["clean_text"], "negation_R3_R4_identical_linear_input": example_results[2]["clean_text"] == example_results[3]["clean_text"]}
    binary = [x for x in example_results if x["expected"] is not None]
    report["examples_binary_exploratory_not_population_estimate"] = metrics([x["expected"] for x in binary], [x["predicted_label"] for x in binary])
    # Warm sequential benchmarks: 30 calls, varying length, no scraping/DB/network.
    e.predict(EMPLOYERS)
    hybrid_samples, linear_samples = [], []
    for i in range(30):
        text = CASES[i % len(CASES)]["text"] if i % 3 else EMPLOYERS * 20
        started = time.perf_counter(); e.predict(text)
        hybrid_samples.append((time.perf_counter()-started)*1000)
        started = time.perf_counter(); e.svc_model.predict_proba([limpar_texto(text)])
        linear_samples.append((time.perf_counter()-started)*1000)
    report["benchmark"].update({"hybrid_warm_sequential": latency_summary(hybrid_samples), "linear_including_preprocessing_warm": latency_summary(linear_samples), "notes": "Mixed short examples and repeated long text; HF truncates to 1500 characters. No concurrency, web extraction, network, cache, or DB latency. Local Apple ARM CPU is not a cloud benchmark.", "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    # Read-only DB evidence. Do not store user votes or article contents.
    for db_name in ["mlflow.db", "fake_news_local.db"]:
        conn = sqlite3.connect(f"file:{ROOT / db_name}?mode=ro", uri=True)
        tables = {x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        selected = ["experiments", "runs", "metrics", "registered_models", "model_versions"] if db_name == "mlflow.db" else ["noticias_cache", "votos_comunidade"]
        report.setdefault("local_db_counts", {})[db_name] = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in selected if t in tables}
        conn.close()
    # Exercise real application startup + health without write endpoints.
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as client:
        response = client.get("/health")
        report["health_with_real_lifespan"] = {"status_code": response.status_code, "body": response.json()}
    save_json(OUT / "metrics.json", report)
    print("BENCHMARK",json.dumps(report["benchmark"]),flush=True)
    print("SAVED",str(OUT),flush=True)


if __name__ == "__main__":
    main()
