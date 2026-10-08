"""Conservative, auditable sentence matching. Not a calibrated fact-checker."""
from collections import Counter
from datetime import date
from decimal import Decimal
import hashlib
import re
import time
import unicodedata
from tld import get_fld

from ml.contracts import Claim, ClaimResult, Evidence, NewsDocument, VerificationResult, Verdict
from ml.preprocessing import compor_texto

SYSTEM_VERSION = "evidence-rules-v0.2.0"
MAX_CLAIMS = 12
UNCERTAINTY = re.compile(r"\b(boato|sarcasmo|sátira|satira|alega|suposto|suposta|talvez|poderia|segundo|disse|afirmou)\b", re.I)
NEGATIONS = {"nao", "nunca", "jamais"}
# Exact structural comparison is deliberately conservative: no synonym guessing.
NUMERIC = re.compile(r"^[+-]?\d+(?:[.,/-]\d+)*$")
FULL_DATE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})$")
VERBAL_DATE = re.compile(r"\b\d{1,2}(?:º)?\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)(?:\s+de\s+\d{4})?\b",re.I)


def tokens(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\br\$|[€$£%‰°]|(?<!\w)[+-]\d+(?:[.,]\d+)*|\b\w+(?:[.,/-]\w+)*\b", text)


def sentences(text):
    # Keep decimal points and dates inside sentences.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def extract_claims(document):
    parts = sentences(compor_texto(document.title, document.body))
    seen, result = set(), []
    for part in parts:
        digest = hashlib.sha256(" ".join(tokens(part)).encode()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        limitations = []
        # Questions, attributed claims, ambiguous pronouns and irony are not asserted facts.
        checkable = len(tokens(part)) >= 4 and not part.endswith("?")
        if UNCERTAINTY.search(part) or re.search(r'["“”]|\b(ele|ela|isso|isto|eles|elas)\b', part, re.I):
            checkable = False
            limitations.append("Atribuição, referência ambígua ou modalidade exige revisão humana.")
        if len(part)>2000 or re.search(r"\b(sátira|satira|sarcasmo)\b", document.body, re.I):
            checkable = False
            limitations.append("Trecho truncado ou artigo com marcador de sátira; revisão obrigatória.")
        if re.search(r"\b(hoje|ontem|amanhã|amanha)\b", part, re.I):
            checkable = False
            limitations.append("Referência temporal relativa ainda não resolvida; revisão obrigatória.")
        if not checkable:
            limitations.append("Extração heurística não confirma uma alegação factual autônoma.")
        event = document.event_date
        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", part)
        local = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", part)
        try:
            if iso: event = date.fromisoformat(iso.group(1))
            elif local: event = date(int(local.group(3)), int(local.group(2)), int(local.group(1)))
        except ValueError:
            limitations.append("Data explícita inválida; revisão temporal necessária.")
            checkable = False
        result.append(Claim(id=digest[:16], text=part[:2000], event_date=event,
                            checkable=checkable, limitations=limitations))
        if len(result) == MAX_CLAIMS: break
    return result, len(seen) >= MAX_CLAIMS or any(len(s)>2000 for s in parts)


def compare(claim_text, excerpt):
    left, right = tokens(claim_text), tokens(excerpt)
    overlap = len(set(left) & set(right)) / max(1, len(set(left) | set(right)))
    if UNCERTAINTY.search(excerpt) or re.search(r'["“”]', excerpt):
        return "nao_conclusiva", overlap
    # Different event dates are different propositions, not numerical refutations.
    if [t for t in left if FULL_DATE.fullmatch(t)] != [t for t in right if FULL_DATE.fullmatch(t)]:
        return "nao_conclusiva", overlap
    if [m.lower() for m in VERBAL_DATE.findall(claim_text)] != [m.lower() for m in VERBAL_DATE.findall(excerpt)]:
        return "nao_conclusiva", overlap
    if re.search(r"\bnão\s+(só|necessariamente|menos|mais)\b",claim_text+" "+excerpt,re.I):
        return "nao_conclusiva", overlap
    def frame(items):
        return [("#" if NUMERIC.fullmatch(t) else t) for t in items if t not in NEGATIONS]
    if frame(left) != frame(right) or len(frame(left)) < 4:
        return "nao_conclusiva", overlap
    nleft = [x for x in left if x in NEGATIONS]
    nright = [x for x in right if x in NEGATIONS]
    numbers_l = [x for x in left if NUMERIC.fullmatch(x)]
    numbers_r = [x for x in right if NUMERIC.fullmatch(x)]
    # Multiple negatives, or differing negation types, may be idiomatic.
    if len(nleft)>1 or len(nright)>1 or (nleft and nright and nleft != nright):
        return "nao_conclusiva", overlap
    if nleft and nright and left.index(nleft[0]) != right.index(nright[0]):
        return "nao_conclusiva", overlap
    if bool(nleft) != bool(nright):
        return "refuta", overlap
    if numbers_l != numbers_r:
        if re.search(r"\b(até|ate|cerca|aproximadamente|entre|superior|inferior|mais|menos)\b", claim_text, re.I):
            return "nao_conclusiva", overlap
        def quantity(value):
            if FULL_DATE.fullmatch(value): return value
            if re.fullmatch(r"[+-]?\d+(?:,\d+)?",value):
                return Decimal(value.replace(",","."))
            if re.fullmatch(r"[+-]?\d+\.\d{1,2}",value): return Decimal(value)
            # Dots with three digits, ranges and fractions have ambiguous meaning.
            return None
        left_values, right_values = [quantity(n) for n in numbers_l], [quantity(n) for n in numbers_r]
        if None in left_values or None in right_values:
            return "nao_conclusiva", overlap
        if left_values == right_values: return "sustenta", overlap
        # Numeric incompatibility with negation is not an entailment.
        return ("nao_conclusiva" if nleft else "refuta"), overlap
    return "sustenta", overlap


def evaluate_claim(claim, document, sources, reference_date):
    evidence = []
    for source in sources:
        # The input itself and verbatim republications cannot independently verify it.
        if document.source_url and source.source_url == document.source_url:
            continue
        matches = []
        for excerpt in sentences(source.body):
            if len(excerpt)>2000: continue
            relation, relevance = compare(claim.text, excerpt)
            if relevance >= .35:
                matches.append((relation != "nao_conclusiva", relevance, excerpt, relation))
        if not matches: continue
        _, relevance, excerpt, relation = max(matches)
        limitations = []
        if not claim.checkable:
            relation = "nao_conclusiva"
        if source.published_at is None:
            relation = "nao_conclusiva"
            limitations.append("Publicação sem data rastreável.")
        elif "ambiguous" in source.date_provenance.get("published_at", ""):
            relation = "nao_conclusiva"
            limitations.append("Data inferida ambígua; não autoriza conclusão temporal.")
        elif source.published_at.date() > reference_date:
            relation = "nao_conclusiva"
            limitations.append("Fonte posterior à data de referência.")
        elif claim.event_date and source.published_at.date() < claim.event_date:
            relation = "nao_conclusiva"
            limitations.append("Fonte antecede o evento alegado.")
        elif not claim.event_date and (reference_date-source.published_at.date()).days>30:
            relation = "nao_conclusiva"
            limitations.append("Fonte antiga sem data do evento explícita.")
        if not source.source_url: continue
        # Identical source content is one origin, even on different domains.
        domain = get_fld(str(source.source_url),fail_silently=True) or source.source_url.host.lower().removeprefix("www.")
        origin = domain
        body_hash = hashlib.sha256(" ".join(tokens(source.body)).encode()).hexdigest()
        evidence.append(Evidence(
            id=hashlib.sha256((str(source.source_url)+excerpt).encode()).hexdigest()[:16],
            claim_id=claim.id, url=source.source_url, title=source.title, excerpt=excerpt,
            published_at=source.published_at, updated_at=source.updated_at,
            date_provenance=source.date_provenance, accessed_at=source.fetched_at,
            content_hash=body_hash, origin_group=origin, provider="download_https",
            relation=relation, relevance_score=relevance, limitations=limitations))
    # Collapse same-domain, exact-content and substantial excerpt copies into one vote.
    votes = {"sustenta":set(), "refuta":set()}
    seen_hashes, seen_domains, voted_texts = set(), set(), []
    for item in evidence:
        # Exact same matching assertion may occur independently; full-body hash detects copies.
        key = item.content_hash
        if key in seen_hashes or item.origin_group in seen_domains: continue
        source = next(s for s in sources if s.source_url == item.url)
        content_tokens = tokens(source.body)
        word_set = set(content_tokens)
        numbers = [t for t in content_tokens if NUMERIC.fullmatch(t)]
        negation = [t for t in content_tokens if t in NEGATIONS]
        if any(len(word_set)>=20 and numbers==old_numbers and negation==old_negation
               and len(word_set & old_words)/max(1,len(word_set|old_words)) >= .90
               for old_words,old_numbers,old_negation in voted_texts):
            item.limitations.append("Texto quase idêntico a outra fonte; não contado como origem independente.")
            continue
        if item.relation in votes:
            votes[item.relation].add(item.origin_group)
            seen_hashes.add(key)
            seen_domains.add(item.origin_group)
            voted_texts.append((word_set,numbers,negation))
    if votes["sustenta"] and votes["refuta"]:
        verdict = Verdict.CONFLICTING
    elif len(votes["sustenta"])>=2:
        verdict = Verdict.SUPPORTED
    elif len(votes["refuta"])>=2:
        verdict = Verdict.REFUTED
    else:
        verdict = Verdict.INSUFFICIENT
    temporal = "evento_explicito" if claim.event_date else "evento_nao_identificado"
    rationale = (
        f"Regra conservadora: {len(votes['sustenta'])} origem(ns) compatível(is), "
        f"{len(votes['refuta'])} incompatível(is); são necessárias duas origens para decisão unilateral. "
        "Relações calculadas sobre os trechos exibidos, não sobre o score textual."
    )
    return ClaimResult(claim=claim, verdict=verdict, evidence=evidence,
        rationale=rationale, temporal_status=temporal,
        limitations=claim.limitations + [
            "Comparação lexical estrita; equivalências semânticas e dependência editorial exigem revisão.",
            "Concordância de fontes não garante verdade; autoridade e contexto precisam ser avaliados.",
        ])


def verify(document: NewsDocument, sources: list[NewsDocument], reference_date: date, *, retrieval_status="provided_urls"):
    start = time.perf_counter()
    claims, truncated = extract_claims(document)
    results = [evaluate_claim(c, document, sources, reference_date) for c in claims]
    counts = Counter(r.verdict.value for r in results)
    return VerificationResult(system_version=SYSTEM_VERSION, document=document,
        reference_date=reference_date, results=results,
        summary={label.value: counts[label.value] for label in Verdict},
        partial=truncated or len(counts)>1,
        limitations=[
            "Protótipo de regras, sem validação externa recente ou confiança factual calibrada.",
            "Metadados e trechos de sites podem estar incorretos; não há autenticação editorial.",
            "Deduplicação por domínio/conteúdo não detecta todas as republicações.",
        ] + (["Texto excede a cobertura de 12 alegações/2000 caracteres por alegação."] if truncated else []),
        retrieval_status=retrieval_status, latency_ms=round((time.perf_counter()-start)*1000,3))
