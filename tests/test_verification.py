from datetime import date, datetime, timezone
import pytest

from ml.contracts import NewsDocument, Verdict
from ml.verification import compare, extract_claims, verify

ASOF = date(2026,10,8)
FACT = "A prefeitura registrou 82 casos em 06/10/2026."


def document(text=FACT):
    return NewsDocument(body=text)


def source(text=FACT, host="a.example", published="2026-10-07", tail="Boletim técnico municipal."):
    return NewsDocument(title="Boletim", body=text+"\n"+tail, source_url=f"https://{host}/boletim",
        published_at=datetime.fromisoformat(published).replace(tzinfo=timezone.utc) if published else None)


@pytest.mark.parametrize("left,right,relation", [
    (FACT,FACT,"sustenta"),
    (FACT,FACT.replace("82","8200"),"refuta"),
    (FACT,FACT.replace("registrou","não registrou"),"refuta"),
    (FACT,FACT.replace("prefeitura","universidade"),"nao_conclusiva"),
    (FACT,"Segundo alguém, "+FACT,"nao_conclusiva"),
    (FACT,FACT.replace("06/10/2026","07/10/2026"),"nao_conclusiva"),
    ("A prefeitura registrou até 82 casos.","A prefeitura registrou até 83 casos.","nao_conclusiva"),
    ("A prefeitura registrou 82 casos em 6 de outubro.","A prefeitura registrou 82 casos em 7 de outubro.","nao_conclusiva"),
    ("A prefeitura registrou 82 casos em 6 de outubro.","A prefeitura registrou 83 casos em 6 de outubro.","refuta"),
    ("Não só a prefeitura registrou 82 casos.","Só a prefeitura não registrou 82 casos.","nao_conclusiva"),
    ("A taxa registrada foi 8,2%.","A taxa registrada foi 8.2%.","sustenta"),
    ("A taxa registrada foi 8,2%.","A taxa registrada foi 9,2%.","refuta"),
    ("O município recebeu 8.200 equipamentos.","O município recebeu 8200 equipamentos.","nao_conclusiva"),
    ("A taxa registrada foi 8,2%.","A taxa registrada foi 8,2.","nao_conclusiva"),
    ("O valor registrado foi R$ 82.","O valor registrado foi € 82.","nao_conclusiva"),
    ("O saldo registrado foi -82 reais.","O saldo registrado foi 82 reais.","refuta"),
    ("A prefeitura não registrou 82 casos.", "A prefeitura não registrou 83 casos.", "nao_conclusiva"),
])
def test_comparison_preserves_fact_frame(left,right,relation):
    assert compare(left,right)[0] == relation


def test_no_evidence_abstains():
    result = verify(document(),[],ASOF)
    assert result.results[0].verdict == Verdict.INSUFFICIENT
    assert result.results[0].confidence is None


def test_two_sources_support_with_provenance():
    result = verify(document(),[source(),source(host="b.example",tail="Relatório estadual independente.")],ASOF)
    row = result.results[0]
    assert row.verdict == Verdict.SUPPORTED
    assert len(row.evidence) == 2
    assert all(e.claim_id == row.claim.id and e.published_at and e.content_hash for e in row.evidence)
    assert row.confidence is None


def test_two_sources_refute_numeric_change():
    result = verify(document(FACT.replace("82","8200")),[source(),source(host="b.example",tail="Relatório independente.")],ASOF)
    assert result.results[0].verdict == Verdict.REFUTED


def test_one_each_is_conflicting_not_majority_binary():
    result = verify(document(),[source(),source(FACT.replace("82","83"),host="b.example")],ASOF)
    assert result.results[0].verdict == Verdict.CONFLICTING


@pytest.mark.parametrize("kind", ["copy","same_domain","self"])
def test_republication_and_self_are_not_independent(kind):
    original = source()
    if kind == "copy": second = source(host="b.example")
    elif kind == "same_domain": second = source(tail="Outro relato.")
    else:
        doc = document()
        doc.source_url = original.source_url
        result = verify(doc,[original],ASOF)
        assert not result.results[0].evidence
        return
    assert verify(document(),[original,second],ASOF).results[0].verdict == Verdict.INSUFFICIENT


def test_subdomains_do_not_vote_as_independent_publishers():
    result=verify(document(),[source(host="news.example.com"),source(host="blog.example.com",tail="Outro relato.")],ASOF)
    assert result.results[0].verdict == Verdict.INSUFFICIENT


def test_near_copy_with_added_byline_is_one_origin():
    tail="Relatório técnico de vigilância municipal elaborado pelas equipes responsáveis após a coleta de dados nas unidades de atendimento e análise dos registros disponíveis nos sistemas administrativos."
    result=verify(document(),[source(tail=tail),source(host="b.example",tail=tail+" Redação.")],ASOF)
    assert result.results[0].verdict == Verdict.INSUFFICIENT


@pytest.mark.parametrize("published", [None, "2026-10-09", "2026-10-01"])
def test_missing_future_or_pre_event_dates_do_not_vote(published):
    result = verify(document(),[source(published=published),source(host="b.example",published=published,tail="Outro relato.")],ASOF)
    assert result.results[0].verdict == Verdict.INSUFFICIENT
    assert all(e.relation=="nao_conclusiva" for e in result.results[0].evidence)


def test_old_fact_can_be_supported_when_event_is_explicit():
    fact = FACT.replace("06/10/2026","06/10/2020")
    result = verify(document(fact),[source(fact,published="2020-10-07"),source(fact,host="b.example",published="2020-10-07",tail="Outro relato.")],ASOF)
    assert result.results[0].verdict == Verdict.SUPPORTED
    assert result.results[0].temporal_status == "evento_explicito"


def test_old_source_without_event_abstains():
    fact = "A prefeitura registrou 82 casos."
    result = verify(document(fact),[source(fact,published="2020-10-07"),source(fact,host="b.example",published="2020-10-07",tail="Outro relato.")],ASOF)
    assert result.results[0].verdict == Verdict.INSUFFICIENT


def test_ambiguous_auto_date_is_not_trusted():
    sources = [source(),source(host="b.example",tail="Outro relato.")]
    for item in sources: item.date_provenance["published_at"] = "auto extraction (ambiguous)"
    result=verify(document(),sources,ASOF)
    assert result.results[0].verdict == Verdict.INSUFFICIENT


def test_partial_article_has_per_claim_results():
    result = verify(document(FACT+"\nA universidade registrou 50 casos."),
                    [source(),source(host="b.example",tail="Outro relato.")],ASOF)
    assert result.partial
    assert [r.verdict for r in result.results] == [Verdict.SUPPORTED,Verdict.INSUFFICIENT]


@pytest.mark.parametrize("text", ["Será que a prefeitura registrou 82 casos?", "Segundo alguém, a prefeitura registrou 82 casos.", "Isso registrou 82 casos.", "Sátira: a prefeitura registrou 82 casos."])
def test_questions_attribution_satire_abstain(text):
    claims, _ = extract_claims(document(text))
    assert not claims[0].checkable


def test_relative_date_requires_resolution():
    claim=extract_claims(document("Hoje a prefeitura registrou 82 casos."))[0][0]
    assert not claim.checkable
