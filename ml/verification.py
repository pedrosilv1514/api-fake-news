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
from ml.sources import registrable_domain, source_category
from ml.temporal import publication_anchor, resolve_event_date
from ml.terms import pertinence

SYSTEM_VERSION = "evidence-rules-v0.3.0"
MAX_CLAIMS = 12
# Minimum claim-term coverage for an excerpt to be listed as pertinent (not a stance).
PERTINENCE_MIN = 0.5
UNCERTAINTY = re.compile(r"\b(boato|sarcasmo|sátira|satira|alega|suposto|suposta|talvez|poderia|segundo|disse|afirmou)\b", re.I)
NEGATIONS = {"nao", "nunca", "jamais"}
# Exact structural comparison is deliberately conservative: no synonym guessing.
NUMERIC = re.compile(r"^[+-]?\d+(?:[.,/-]\d+)*$")
FULL_DATE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})$")
VERBAL_DATE = re.compile(r"\b\d{1,2}(?:º)?\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)(?:\s+de\s+\d{4})?\b",re.I)
# Patterns observed in real extracted pages (08/10/2026 Santos collection).
CALL_TO_ACTION = re.compile(r"^(conheça|confira|veja|leia|acesse|clique|saiba|assista|siga|inscreva-se|baixe|compartilhe|acompanhe)\b", re.I)
DANGLING_END = re.compile(r"\b(em|no|na|nos|nas|de|do|da|dos|das|para|pelo|pela|por|com|a|o|e|ao|à|site|link|aqui)\s*$", re.I)
DEMONSTRATIVE_START = re.compile(r"^(est[ea]s?|ess[ea]s?|aquel[ea]s?|isso|isto|aquilo|tal)\b(?!\s+(semana|mês|mes|ano|segunda|terça|terca|quarta|quinta|sexta|sábado|sabado|domingo|manhã|manha|tarde|noite))", re.I)
FUTURE = re.compile(r"\b(será|serão|vai|vão|irá|irão|deverá|deverão|pretende|pretendem|planeja|planejam)\b", re.I)


def tokens(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\br\$|[€$£%‰°]|(?<!\w)[+-]\d+(?:[.,]\d+)*|\b\w+(?:[.,/-]\w+)*\b", text)


def sentences(text):
    # Keep decimal points and dates inside sentences.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def extract_claims(document):
    parts = sentences(compor_texto(document.title, document.body))
    anchor, anchor_provenance = publication_anchor(
        document.published_at, document.date_provenance.get("published_at"))
    seen, result = set(), []
    for index, part in enumerate(parts):
        digest = hashlib.sha256(" ".join(tokens(part)).encode()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        limitations = []
        # Questions, attributed claims, ambiguous pronouns and irony are not asserted facts.
        checkable = len(tokens(part)) >= 4 and not part.endswith("?")
        if UNCERTAINTY.search(part) or re.search(r'[\"“”]|\b(ele|ela|isso|isto|eles|elas)\b', part, re.I):
            checkable = False
            limitations.append("Atribuição, referência ambígua ou modalidade exige revisão humana.")
        if DEMONSTRATIVE_START.search(part):
            checkable = False
            limitations.append("Referência anafórica ao trecho anterior; contexto anexado para revisão.")
        if CALL_TO_ACTION.search(part):
            checkable = False
            limitations.append("Chamada editorial ou de navegação, não alegação factual.")
        if not re.search(r"[.!?…:)\"”']$", part) and DANGLING_END.search(part):
            checkable = False
            limitations.append("Trecho truncado na extração (link ou elemento removido).")
        if FUTURE.search(part):
            checkable = False
            limitations.append("Previsão ou agenda futura: verificável apenas como anúncio, não como fato ocorrido.")
        if len(part)>2000 or re.search(r"\b(sátira|satira|sarcasmo)\b", document.body, re.I):
            checkable = False
            limitations.append("Trecho truncado ou artigo com marcador de sátira; revisão obrigatória.")
        temporal = resolve_event_date(part, anchor, anchor_provenance)
        limitations.extend(temporal.limitations)
        if temporal.unresolved:
            checkable = False
        event, provenance = temporal.event_date, temporal.provenance
        if event is None and document.event_date and not temporal.unresolved:
            event, provenance = document.event_date, "data do evento informada no documento"
        if not checkable:
            limitations.append("Extração heurística não confirma uma alegação factual autônoma.")
        previous = parts[index-1] if index and parts[index-1] != document.title else ""
        context = " | ".join(x for x in [document.title if index else "", previous] if x)
        result.append(Claim(id=digest[:16], text=part[:2000], event_date=event,
                            event_date_provenance=provenance, context=context[:500],
                            checkable=checkable, limitations=list(dict.fromkeys(limitations))))
        if len(result) == MAX_CLAIMS: break
    return result, len(seen) >= MAX_CLAIMS or any(len(s)>2000 for s in parts)


def compare(claim_text, excerpt):
    left, right = tokens(claim_text), tokens(excerpt)
    from ml.terms import content_terms, anchor_terms
    left_ct, right_ct = content_terms(claim_text), content_terms(excerpt)
    
    if UNCERTAINTY.search(excerpt) or re.search(r'["“”]', excerpt):
        return "nao_conclusiva", 0.0

    if not left_ct: 
        return "nao_conclusiva", 0.0
    coverage = len(left_ct & right_ct) / len(left_ct)
    
    # Require critical entity anchors (non-numeric) to be present in the excerpt
    anchors_l = {a for a in anchor_terms(claim_text) if not NUMERIC.fullmatch(a)}
    if anchors_l:
        anchor_coverage = len(anchors_l & right_ct) / len(anchors_l)
        if anchor_coverage < 0.50:
            return "nao_conclusiva", coverage
            
    if coverage < 0.50:
        return "nao_conclusiva", coverage
        
    left_alpha = {w for w in left_ct if not NUMERIC.fullmatch(w)}
    right_alpha = {w for w in right_ct if not NUMERIC.fullmatch(w)}
    if left_alpha:
        alpha_coverage = len(left_alpha & right_alpha) / len(left_alpha)
        if alpha_coverage < 0.50:
            return "nao_conclusiva", coverage
        
    # Different explicit event dates denote different propositions
    if [t for t in left if FULL_DATE.fullmatch(t)] != [t for t in right if FULL_DATE.fullmatch(t)]:
        return "nao_conclusiva", coverage
    if [m.lower() for m in VERBAL_DATE.findall(claim_text)] != [m.lower() for m in VERBAL_DATE.findall(excerpt)]:
        return "nao_conclusiva", coverage
        
    SYMBOLS = frozenset({"r$", "€", "$", "£", "%", "‰", "°"})
    if [x for x in left if x in SYMBOLS] != [x for x in right if x in SYMBOLS]:
        return "nao_conclusiva", coverage
        
    nleft = [x for x in left if x in NEGATIONS]
    nright = [x for x in right if x in NEGATIONS]
    
    numbers_l = [x for x in left if NUMERIC.fullmatch(x)]
    numbers_r = [x for x in right if NUMERIC.fullmatch(x)]
    
    # Multiple negations in a single sentence often mean idiomatic usage
    if len(nleft) > 1 or len(nright) > 1 or (nleft and nright and nleft != nright):
        return "nao_conclusiva", coverage
        
    if nleft and nright and left.index(nleft[0]) != right.index(nright[0]):
        return "nao_conclusiva", coverage
        
    if numbers_l != numbers_r:
        if re.search(r"\b(até|ate|cerca|aproximadamente|entre|superior|inferior|mais|menos)\b", claim_text, re.I):
            return "nao_conclusiva", coverage
        def quantity(value):
            if FULL_DATE.fullmatch(value): return value
            if re.fullmatch(r"[+-]?\d+(?:,\d+)?",value):
                return Decimal(value.replace(",","."))
            if re.fullmatch(r"[+-]?\d+\.\d{1,2}",value): return Decimal(value)
            return None
        left_values, right_values = [quantity(n) for n in numbers_l], [quantity(n) for n in numbers_r]
        if None in left_values or None in right_values:
            return "nao_conclusiva", coverage
            
        # Every numeric quantity claimed must be found in the excerpt (order-independent)
        if not all(v in right_values for v in left_values):
            # If a quantity differs, and the claim was negative, it might be a refutation, but usually it's just inconclusive
            return ("nao_conclusiva" if nleft else "refuta"), coverage
            
    # If all entities, numbers and dates match, we check semantic polarity (negation)
    if bool(nleft) != bool(nright):
        return "refuta", coverage
        
    return "sustenta", coverage


def evaluate_claim(claim, document, sources, reference_date, provenance=None):
    provenance = provenance or {}
    evidence = []
    document_domain = registrable_domain(document.source_url)
    for source in sources:
        # The input itself and verbatim republications cannot independently verify it.
        if document.source_url and source.source_url == document.source_url:
            continue
        if not source.source_url: continue
        matches = []
        candidates = ([source.title] if source.title else []) + sentences(source.body)
        for excerpt in dict.fromkeys(candidates):
            if len(excerpt)>2000: continue
            relation, overlap = compare(claim.text, excerpt)
            score = pertinence(claim.text, excerpt)
            # Pertinence lists the excerpt; only the strict comparator assigns a stance.
            if overlap >= .35 or score >= PERTINENCE_MIN:
                matches.append((relation != "nao_conclusiva", score, excerpt, relation))
        if not matches: continue
        _, relevance, excerpt, relation = max(matches)
        limitations = []
        if not claim.checkable:
            relation = "nao_conclusiva"
        if source.published_at is None:
            relation = "nao_conclusiva"
            limitations.append("Publicação sem data rastreável.")
        elif "ambiguous" in source.date_provenance.get("published_at", ""):
            # Não invalidar a relação inteira só por data ambígua (melhora UX de 50% constante)
            limitations.append("Data inferida ambígua; não garante cronologia estrita.")
        elif source.published_at.date() > reference_date:
            relation = "nao_conclusiva"
            limitations.append("Fonte posterior à data de referência.")
        elif claim.event_date and source.published_at.date() < claim.event_date:
            relation = "nao_conclusiva"
            limitations.append("Fonte antecede o evento alegado.")
        elif not claim.event_date and (reference_date-source.published_at.date()).days>30:
            relation = "nao_conclusiva"
            limitations.append("Fonte antiga sem data do evento explícita.")
        # Same publisher as the analysed article: shown for context, never an independent vote.
        origin = registrable_domain(source.source_url)
        if document_domain and origin == document_domain:
            relation = "nao_conclusiva"
            limitations.append("Mesmo publicador do texto analisado; não é confirmação independente.")
        found = provenance.get(str(source.source_url), {})
        # Identical source content is one origin, even on different domains.
        body_hash = hashlib.sha256(" ".join(tokens(source.body)).encode()).hexdigest()
        evidence.append(Evidence(
            id=hashlib.sha256((str(source.source_url)+excerpt).encode()).hexdigest()[:16],
            claim_id=claim.id, url=source.source_url, title=source.title, excerpt=excerpt,
            published_at=source.published_at, updated_at=source.updated_at,
            date_provenance=source.date_provenance, accessed_at=source.fetched_at,
            content_hash=body_hash, origin_group=origin,
            provider=str(found.get("provider") or "download_https"),
            source_category=source_category(source.source_url), retrieval=dict(found),
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


def verify(document: NewsDocument, sources: list[NewsDocument], reference_date: date, *, retrieval_status="provided_urls", provenance=None):
    start = time.perf_counter()
    claims, truncated = extract_claims(document)
    results = [evaluate_claim(c, document, sources, reference_date, provenance) for c in claims]
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
