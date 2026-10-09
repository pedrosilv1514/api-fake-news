"""Event-date resolution for claims. Publication date is an anchor, never the event itself.

Only patterns whose meaning is unambiguous given a reliable anchor are resolved; every
resolution states its provenance. Anything else stays unresolved and blocks decisions.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from zoneinfo import ZoneInfo

from ml.terms import fold

MONTHS = {fold(m): i for i, m in enumerate([
    "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
    "agosto", "setembro", "outubro", "novembro", "dezembro"], 1)}
WEEKDAYS = {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4, "sabado": 5, "domingo": 6}
ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
LOCAL = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
VERBAL = re.compile(r"\b(\d{1,2})(?:º|o)?\s+de\s+(" + "|".join(MONTHS) + r")(?:\s+de\s+(\d{4}))?\b")
WEEKDAY_DAY = re.compile(r"\b(segunda|terca|quarta|quinta|sexta)(?:-feira)?\s*\((\d{1,2})\)|\b(sabado|domingo)\s*\((\d{1,2})\)")
WEEKDAY_ONLY = re.compile(r"\b(segunda|terca|quarta|quinta|sexta)-feira\b|\b(sabado|domingo)\b")
RELATIVE = {"anteontem": -2, "ontem": -1, "hoje": 0, "amanha": 1}
RELATIVE_RE = re.compile(r"\b(anteontem|ontem|hoje|amanha)\b")
NEWS_ZONE = ZoneInfo("America/Sao_Paulo")


@dataclass
class TemporalResolution:
    event_date: date | None = None
    provenance: str | None = None
    unresolved: bool = False
    limitations: list[str] = field(default_factory=list)


def publication_anchor(published_at: datetime | None, provenance: str | None) -> tuple[date | None, str]:
    """Local publication day usable for relative references, or None if unreliable."""
    if published_at is None:
        return None, "publicação sem data"
    provenance = provenance or "data de publicação informada na entrada"
    if "ambiguous" in provenance:
        return None, "data de publicação ambígua"
    if "date precision" in provenance:
        # Day-precision labels are stored at 00:00 UTC; converting them would shift a day.
        return published_at.date(), provenance
    # Timestamped metadata: the newsroom's local day (assumed Brazil for pt-BR news).
    return published_at.astimezone(NEWS_ZONE).date(), provenance + "; dia local America/Sao_Paulo"


def _nearest(candidates, anchor):
    valid = [c for c in candidates if c is not None]
    return min(valid, key=lambda c: abs((c - anchor).days)) if valid else None


def _safe_date(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def resolve_event_date(text: str, anchor: date | None, anchor_provenance: str = "") -> TemporalResolution:
    folded = fold(text)
    found: list[tuple[date, str]] = []
    result = TemporalResolution()

    for match in ISO.finditer(folded):
        d = _safe_date(int(match[1]), int(match[2]), int(match[3]))
        if d is None:
            result.unresolved = True
            result.limitations.append("Data explícita inválida; revisão temporal necessária.")
        else:
            found.append((d, "data explícita no texto"))
    for match in LOCAL.finditer(folded):
        d = _safe_date(int(match[3]), int(match[2]), int(match[1]))
        if d is None:
            result.unresolved = True
            result.limitations.append("Data explícita inválida; revisão temporal necessária.")
        else:
            found.append((d, "data explícita no texto"))
    for match in VERBAL.finditer(folded):
        day, month = int(match[1]), MONTHS[match[2]]
        if match[3]:
            d = _safe_date(int(match[3]), month, day)
            if d: found.append((d, "data explícita no texto"))
            continue
        if anchor is None:
            result.unresolved = True
            result.limitations.append("Dia/mês sem ano e sem data de publicação confiável.")
            continue
        d = _nearest([_safe_date(anchor.year + k, month, day) for k in (-1, 0, 1)], anchor)
        if d: found.append((d, "dia/mês explícitos; ano inferido da publicação"))
    for match in WEEKDAY_DAY.finditer(folded):
        weekday = WEEKDAYS[match[1] or match[3]]
        day = int(match[2] or match[4])
        if anchor is None:
            result.unresolved = True
            result.limitations.append("Dia da semana sem data de publicação confiável.")
            continue
        candidates = []
        for offset in (-1, 0, 1):
            year, month = anchor.year, anchor.month + offset
            if month == 0: year, month = year - 1, 12
            if month == 13: year, month = year + 1, 1
            candidates.append(_safe_date(year, month, day))
        d = _nearest([c for c in candidates if c and c.weekday() == weekday], anchor)
        if d is None:
            result.unresolved = True
            result.limitations.append("Dia da semana incompatível com o dia do mês informado.")
        else:
            found.append((d, "dia da semana e dia do mês; mês/ano inferidos da publicação"))
    # Weekday names without a day number are ambiguous ("nesta segunda" vs. next week).
    weekday_spans = [m.span() for m in WEEKDAY_DAY.finditer(folded)]
    for match in WEEKDAY_ONLY.finditer(folded):
        if not any(start <= match.start() < end for start, end in weekday_spans):
            result.unresolved = True
            result.limitations.append("Dia da semana sem dia do mês; data do evento ambígua.")
            break
    for match in RELATIVE_RE.finditer(folded):
        if anchor is None:
            result.unresolved = True
            result.limitations.append("Referência temporal relativa sem data de publicação confiável.")
            continue
        found.append((anchor + timedelta(days=RELATIVE[match[1]]), "referência relativa resolvida pela publicação"))

    distinct = sorted({d for d, _ in found})
    past = [d for d in distinct if anchor is None or d <= anchor]
    future = [d for d in distinct if anchor is not None and d > anchor]
    if future:
        # Scheduled/effective dates (agenda, vigência) are not the reported event's date.
        result.limitations.append("Data futura mencionada (agenda/vigência); não é tratada como data do evento.")
    if len(past) == 1 and not result.unresolved:
        result.event_date = past[0]
        source = next(p for d, p in found if d == past[0])
        result.provenance = source + (f" (âncora: {anchor_provenance})" if "publica" in source and anchor_provenance else "")
    elif len(past) > 1:
        result.limitations.append("Várias datas passadas na mesma frase; data do evento exige revisão.")
    return result
