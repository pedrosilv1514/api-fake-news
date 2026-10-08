import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ml.candidate import CandidateClassifier
from ml.contracts import NewsDocument
from ml.dataset import load_articles, temporal_splits
from ml.preprocessing import compor_texto, normalizar_texto

ROOT = Path(__file__).resolve().parents[1]


def test_preserve_negation_numbers_units_dates():
    text = normalizar_texto("Não houve 82 casos: 8,2% em 06/10/2026 e 82 kg.")
    for token in ["não", "82", "8,2%", "06/10/2026", "kg"]:
        assert token in text
    assert text != normalizar_texto("Houve 8200 casos: 82% em 07/10/2026 e 82 kg.")


def test_training_serving_compose_is_same():
    assert compor_texto("Título", "Corpo") == "Título\nCorpo"
    assert compor_texto("Título", "Título\nCorpo") == "Título\nCorpo"
    assert compor_texto("", "Corpo") == "Corpo"


def test_contract_hash_ignores_access_not_dates():
    first = NewsDocument(body="Não houve 82 casos.")
    second = NewsDocument(body="Não houve 82 casos.")
    assert first.content_hash() == second.content_hash()
    second.body = "Houve 8200 casos."
    assert first.content_hash() != second.content_hash()


def test_dataset_rejects_cleaned_only_and_unreviewed(tmp_path):
    path = tmp_path/"input.csv"
    pd.DataFrame({"texto_limpo":["texto"], "Classe":[0]}).to_csv(path,index=False)
    with pytest.raises(ValueError, match="colunas"): load_articles(path)
    pd.DataFrame({"Titulo":["t"],"Noticia":["b"],"Data":["01/01/2020"],"URL":["https://a.example"],"Classe":[-1]}).to_csv(path,index=False)
    with pytest.raises(ValueError, match="feedback"): load_articles(path)


def test_temporal_splits_purge_boundary_group():
    rows = [{"group_id":f"g{i}", "published":pd.Timestamp(f"2020-01-{i:02}"),"Classe":c}
            for i in range(1,21) for c in [0,1]]
    # A related story straddling the first boundary must never leak.
    rows += [{"group_id":"cross", "published":pd.Timestamp("2020-01-01"),"Classe":0},
             {"group_id":"cross", "published":pd.Timestamp("2020-01-20"),"Classe":1}]
    parts, info = temporal_splits(pd.DataFrame(rows))
    assert info["purged_rows"] == 2
    assert parts["train"].published.max() < parts["calibration"].published.min()
    assert parts["calibration"].published.max() < parts["test"].published.min()
    assert all("cross" not in set(part.group_id) for part in parts.values())


def candidate_dir():
    return sorted((ROOT/"models/candidates").glob("candidate-*"))[-1]


def test_real_candidate_matches_raw_pipeline_and_manifest():
    directory = candidate_dir()
    classifier = CandidateClassifier(directory)
    doc = NewsDocument(title="Prefeitura publica boletim", body="Não houve 82 casos em 06/10/2026.")
    expected = classifier.model.predict_proba([compor_texto(doc.title,doc.body)])[0][0]
    assert classifier.predict(doc)["score_textual_fake"] == expected
    assert classifier.predict(doc)["confianca_factual"] is None
    assert not classifier.predict(doc)["production_approved"]
    vectorizer = classifier.model.calibrated_classifiers_[0].estimator.named_steps["tfidf"]
    tokens = vectorizer.build_analyzer()("Não houve 8,2% em 06/10/2026; 82 kg e R$ 82.")
    assert all(t in tokens for t in ["não", "8,2%", "06/10/2026", "kg"])


def test_candidate_integrity_rejects_tampering(tmp_path):
    directory = candidate_dir()
    manifest = json.loads((directory/"manifest.json").read_text())
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"model.joblib").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="Hash"): CandidateClassifier(tmp_path)


def test_candidate_actual_split_metadata_is_disjoint():
    directory = candidate_dir()
    parts = {n:pd.read_csv(directory/f"{n}_split.csv") for n in ["train","calibration","test"]}
    for l,r in [("train","calibration"),("train","test"),("calibration","test")]:
        for col in ["article_id","group_id","text_hash"]:
            assert not set(parts[l][col]) & set(parts[r][col])
        assert pd.to_datetime(parts[l].published).max() < pd.to_datetime(parts[r].published).min()


def test_legacy_xai_direction_matches_local_probability_gradient():
    from ml.inference import NLPEngine
    engine = NLPEngine()
    engine.load_model()
    assert engine.ready
    prediction = engine.predict("O governo não registrou 82 casos de fraude.")
    assert prediction["confianca_factual"] is None
    xai = engine.xai_engine
    # Calibration, not assumptions about base class semantics, determines direction.
    a = xai.svc_model.calibrated_classifiers_[0].calibrators[0].a_
    assert a == xai.fake_direction
    for item in prediction["explicacao_xai"]["contribuicoes_textuais"]:
        idx = list(xai.feature_names).index(item["termo"])
        assert (item["direcao_fake"] > 0) == (xai.coefs[idx]*a > 0)


def test_xai_domain_is_not_substring_trusted():
    from ml.inference import NLPEngine
    engine = NLPEngine()
    engine.load_model()
    explanation = engine.predict("O governo registrou casos.", "https://gov.br.evil.example/a")["explicacao_xai"]
    assert "requer verificação adicional" in explanation["janela_analise"]["qualidade_da_fonte"]
    assert "Não verificada" in explanation["janela_analise"]["corroboracao"]


def test_missing_legacy_assets_fail_closed_without_zero_shot_factual_fallback(monkeypatch):
    from ml.inference import NLPEngine
    def absent(text): raise RuntimeError("Stopwords ausentes")
    monkeypatch.setattr("ml.inference.limpar_texto",absent)
    engine=NLPEngine()
    engine.load_model()
    assert not engine.ready
    with pytest.raises(RuntimeError,match="indisponível"):
        engine.predict("Texto para analisar.")
