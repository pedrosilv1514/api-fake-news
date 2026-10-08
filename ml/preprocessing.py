"""Versioned preprocessing: preserve facts for new models; keep legacy serving stable."""
import re
import unicodedata
from functools import lru_cache

PREPROCESSING_VERSION = "facts-v2"
LEGACY_PREPROCESSING_VERSION = "legacy-v1"


def normalizar_texto(texto: str) -> str:
    """Preserve negation, quantities, units and dates; no downloads or stemming."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFC", texto).lower()
    texto = re.sub(r"https?://\S+|www\.\S+", " urltoken ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def compor_texto(titulo: str, corpo: str) -> str:
    """The same title/body contract is used inside training and new inference."""
    titulo, corpo = titulo.strip(), corpo.strip()
    return f"{titulo}\n{corpo}".strip() if titulo and not corpo.startswith(titulo) else corpo or titulo


@lru_cache(maxsize=1)
def _legacy_stopwords():
    from nltk.corpus import stopwords
    try:
        return set(stopwords.words("portuguese"))
    except LookupError as exc:
        raise RuntimeError("NLTK stopwords ausentes. Instale-as no build; runtime não faz downloads.") from exc


def limpar_texto(texto: str) -> str:
    """Only for the frozen legacy artifact. Do not use in new training."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"https?://\S+|www\.\S+", " urltoken ", texto.lower())
    texto = re.sub(r"[^a-zA-Záàâãéèêíïóôõöúçñ\s]", " ", texto)
    stopwords = _legacy_stopwords()
    return " ".join(w for w in texto.split() if w not in stopwords and len(w) > 2)
