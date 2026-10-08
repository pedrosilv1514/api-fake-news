import pytest

from scraper.extractor import ExtractionError, extract_document
from scraper.safe_http import FetchedPage, FetchError

HTML = """<html lang="pt"><head><title>Boletim de saúde</title>
<meta property="article:published_time" content="2026-10-07T10:00:00Z"/></head>
<body><article><h1>Boletim de saúde</h1>
<p>A prefeitura não registrou 82 casos em 06/10/2026. O relatório oficial descreve
os dados coletados nas unidades de saúde do município e apresenta a comparação
com os boletins anteriores. As equipes de vigilância acompanharam os registros.</p>
<p>A divulgação contém informações sobre 8,2% dos atendimentos e 82 kg de material
analisado. O levantamento reúne dados atualizados e critérios de acompanhamento
apresentados pelas equipes responsáveis pelo boletim.</p></article></body></html>"""


def test_extract_document_fixture_preserves_fact_and_provenance():
    doc=extract_document("https://example.com/news",fetcher=lambda u:FetchedPage(u,HTML.encode(),"text/html"))
    assert "não" in doc.body and "82" in doc.body and "06/10/2026" in doc.body
    assert "8,2%" in doc.body
    assert doc.title=="Boletim de saúde"
    assert doc.published_at.date().isoformat()=="2026-10-07"
    assert str(doc.source_url)=="https://example.com/news"


def test_extraction_failure_is_asserted_not_swallowed():
    def unavailable(url): raise FetchError("DNS indisponível")
    with pytest.raises(ExtractionError,match="DNS"):
        extract_document("https://example.com",fetcher=unavailable)


def test_visible_publication_outweighs_related_article_date():
    markup = HTML.replace('<meta property="article:published_time" content="2026-10-07T10:00:00Z"/>','')
    markup = markup.replace('<h1>Boletim de saúde</h1>',
        '<h1>Boletim de saúde</h1><p>Publicado: 7 de outubro de 2026 - 15h58</p>'
        '<p>Atualizado: 8 de outubro de 2026 - 14h05</p>')
    markup = markup.replace('</body>','<aside>06/10/2026 Outra notícia relacionada</aside></body>')
    doc=extract_document("https://example.com/news",fetcher=lambda u:FetchedPage(u,markup.encode(),"text/html"))
    assert doc.published_at.date().isoformat()=="2026-10-07"
    assert doc.updated_at.date().isoformat()=="2026-10-08"
    assert "date precision" in doc.date_provenance["published_at"]


def test_invalid_structured_date_does_not_guess():
    from scraper.extractor import _date
    assert _date("nonsense") is None
    assert _date(None) is None
