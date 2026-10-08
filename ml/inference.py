"""Auxiliary classifiers. No classifier score is a factual verdict."""
import logging
import os
from pathlib import Path

import joblib

from ml.preprocessing import limpar_texto
from ml.xai import ExplainabilityEngine

logger = logging.getLogger(__name__)


class NLPEngine:
    def __init__(self):
        self.hf_classifier = None
        self.hf_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.hf_revision = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
        self.svc_model = None
        self.xai_engine = None
        self.load_error = None

    @property
    def ready(self):
        return self.svc_model is not None

    def load_model(self):
        self.svc_model = None
        self.xai_engine = None
        self.hf_classifier = None
        self.load_error = None
        try:
            self.svc_model = joblib.load(
                Path(__file__).resolve().parents[1] / "models/calibrated_pipeline.joblib"
            )
            if set(self.svc_model.classes_) != {0, 1}:
                raise ValueError("Semântica inesperada: 0=FAKE, 1=REAL")
            # Fail readiness if the legacy preprocessor assets are unavailable.
            limpar_texto("Teste não contém 82 casos.")
            self.xai_engine = ExplainabilityEngine(self.svc_model)
        except Exception as exc:
            self.svc_model = None
            self.load_error = type(exc).__name__
            logger.exception("Classificador textual indisponível; sem fallback factual")
        # Optional and offline by default. Startup never downloads models.
        if os.getenv("ENABLE_BIAS_MODEL", "0") == "1":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, revision=self.hf_revision, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name, revision=self.hf_revision, local_files_only=True)
                self.hf_classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
            except Exception:
                logger.exception("Análise auxiliar de tom indisponível; checagem não depende dela")

    def predict(self, text: str, url: str = "") -> dict:
        if not self.ready:
            raise RuntimeError("Classificador textual indisponível")
        cleaned = limpar_texto(text)
        fake_index = list(self.svc_model.classes_).index(0)
        prob_fake = float(self.svc_model.predict_proba([cleaned])[0][fake_index])
        bias_label, bias_score, bias_short = "indisponível", None, "UNKNOWN"
        used = ["LinearSVC (BR; referência histórica)"]
        if self.hf_classifier is not None:
            labels = ["neutro, imparcial, informativo", "opinativo, enviesado, sensacionalista"]
            prediction = self.hf_classifier(text[:1500], labels, multi_label=False)
            bias_label = prediction["labels"][0].split(",")[0]
            bias_score = round(float(prediction["scores"][0]), 4)
            bias_short = "NEUTRAL" if prediction["labels"][0] == labels[0] else "BIASED"
            used.append("mDeBERTa-v3 (tom; opcional)")
        return {
            "prob_fake": round(prob_fake, 4),
            "score_kind": "similaridade_textual_calibrada_no_corpus_historico",
            "confianca_factual": None,
            "bias_label": bias_label,
            "bias_score": bias_score,
            "explicacao_xai": self.xai_engine.explain(cleaned, url, prob_fake, bias_short) if self.xai_engine else None,
            "modelos_usados": used,
        }


engine = NLPEngine()
