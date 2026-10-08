"""Opt-in candidate inference with manifest integrity; never promotes artifacts."""
import hashlib
import json
from pathlib import Path

import joblib

from ml.contracts import LABELS, NewsDocument
from ml.preprocessing import PREPROCESSING_VERSION, compor_texto


class CandidateClassifier:
    def __init__(self, directory: Path):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        artifact = directory / "model.joblib"
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != self.manifest["model_sha256"]:
            raise ValueError("Hash do candidato diverge do manifesto")
        if self.manifest["preprocessing_version"] != PREPROCESSING_VERSION:
            raise ValueError("Versão de representação incompatível")
        if self.manifest["label_mapping"] != {str(k): v for k, v in LABELS.items()}:
            raise ValueError("Semântica de rótulos incompatível")
        # Only trusted, locally generated artifacts may be loaded (pickle).
        self.model = joblib.load(artifact)

    def predict(self, document: NewsDocument):
        text = compor_texto(document.title, document.body)
        idx = list(self.model.classes_).index(0)
        return {
            "model_version": self.manifest["version"],
            "score_textual_fake": float(self.model.predict_proba([text])[0][idx]),
            "score_kind": "similaridade_textual_calibrada_no_corpus_historico",
            "confianca_factual": None,
            "production_approved": False,
        }
