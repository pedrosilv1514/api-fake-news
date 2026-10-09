"""Lexical term utilities for retrieval and excerpt pertinence.

Pertinence answers "does this excerpt talk about the claim?" and only decides which
excerpts are shown/ranked. Support or refutation is decided elsewhere, by strict rules.
"""
import re
import unicodedata

# Function words only; content words, negations and quantities are kept.
STOPWORDS = frozenset("""
a o as os um uma uns umas de do da dos das d em no na nos nas num numa nuns numas
por pelo pela pelos pelas para pra pro ao aos à às com sem sob sobre entre ate até
e ou mas que se como quando onde quem qual quais cujo cuja porque pois porem porém
ja já ainda tambem também so só mais menos muito muita muitos muitas pouco
seu sua seus suas meu minha nosso nossa lhe lhes nele nela neles nelas dele dela deles delas
este esta estes estas esse essa esses essas isto isso aquilo aquele aquela
ser e é foi foram era sao são esta está estao estão estava sera será serao serão
ter tem têm tinha teve havia ha há houve
apos após antes depois durante desde contra conforme segundo
the of and
""".split())
NEGATIONS = frozenset({"nao", "não", "nunca", "jamais", "nenhum", "nenhuma"})
NUMBER = re.compile(r"^[+-]?\d+(?:[.,/-]\d+)*$")
WORD = re.compile(r"r\$|[€$£%‰°]|[+-]?\d+(?:[.,/-]\d+)*|[^\W\d_]+(?:-[^\W\d_]+)*", re.U)
SYMBOLS = frozenset({"r$", "€", "$", "£", "%", "‰", "°"})
CAPITALIZED = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]*(?:-[\wÀ-ÿ]+)*|\b[A-Z]{2,}\b")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


FOLDED_STOPWORDS = frozenset(fold(w) for w in STOPWORDS)


def _stem(token: str) -> str:
    if NUMBER.fullmatch(token):
        # 8,2 and 8.2 (two decimals at most) are the same quantity; 8.200 is ambiguous.
        return token.replace(",", ".") if re.fullmatch(r"[+-]?\d+,\d{1,2}", token) else token
    # Five-character prefix: crude Portuguese inflection folding (vistoria/vistorias).
    return token[:5] if len(token) > 5 else token


def content_terms(text: str) -> set[str]:
    terms = set()
    for token in WORD.findall(fold(text)):
        if token in FOLDED_STOPWORDS or (len(token) < 3 and not NUMBER.fullmatch(token) and token not in SYMBOLS):
            continue
        terms.add(_stem(token))
    return terms


def anchor_terms(text: str) -> set[str]:
    """Numbers and named entities (capitalised words not opening the sentence)."""
    anchors = {_stem(t) for t in WORD.findall(fold(text)) if NUMBER.fullmatch(t)}
    stripped = text.strip()
    first = re.match(r"\W*([\wÀ-ÿ-]+)", stripped)
    for match in CAPITALIZED.finditer(stripped):
        if first and match.start() == first.start(1) and not match.group().isupper():
            continue
        for token in WORD.findall(fold(match.group())):
            if token not in FOLDED_STOPWORDS and len(token) >= 2:
                anchors.add(_stem(token))
    return anchors


def pertinence(claim: str, excerpt: str) -> float:
    """Share of claim terms (and of its anchors) present in the excerpt, in [0, 1]."""
    claim_terms = content_terms(claim)
    if not claim_terms:
        return 0.0
    excerpt_terms = content_terms(excerpt)
    coverage = len(claim_terms & excerpt_terms) / len(claim_terms)
    anchors = anchor_terms(claim) & claim_terms
    if not anchors:
        return round(coverage, 4)
    anchor_coverage = len(anchors & excerpt_terms) / len(anchors)
    return round((coverage + anchor_coverage) / 2, 4)


def build_query(claim_text: str, context: str = "", max_terms: int = 12) -> str:
    """Keyword query in original spelling; numbers/entities first, no invented words."""
    words, seen = [], set()

    def add(token):
        key = fold(token)
        if key in seen or key in FOLDED_STOPWORDS or (len(token) < 3 and not NUMBER.fullmatch(token)):
            return
        seen.add(key)
        words.append(token)

    raw = WORD.findall(claim_text)
    anchors = {fold(t) for t in CAPITALIZED.findall(claim_text)} | {t for t in raw if NUMBER.fullmatch(t)}
    for token in raw:
        if fold(token) in anchors or token in anchors:
            add(token)
    for token in raw:
        add(token)
    # Context only contributes named entities, so anaphoric claims gain a subject.
    for token in CAPITALIZED.findall(context):
        if len(words) >= max_terms:
            break
        add(token)
    return " ".join(words[:max_terms])
