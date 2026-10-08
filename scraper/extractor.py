import json
import re
from datetime import datetime, timezone

import trafilatura
from lxml import html

from ml.contracts import NewsDocument
from scraper.safe_http import FetchError, fetch_public
from .sanitizer import sanitize_text


class ExtractionError(Exception):
    pass


def _date(value):
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z","+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def visible_date(tree, label):
    # Labels attached to the article outrank dates in related-news links.
    text = " ".join(tree.itertext())
    months = {name:i for i,name in enumerate([
        "janeiro","fevereiro","março","abril","maio","junho","julho",
        "agosto","setembro","outubro","novembro","dezembro"],1)}
    match = re.search(rf"\b{label}\s*:\s*(\d{{1,2}})\s+de\s+(\w+)\s+de\s+(\d{{4}})", text, re.I)
    if not match: return None
    try:
        # Timezone is not asserted by the page: persist date precision, not invented time.
        return datetime(int(match[3]),months[match[2].lower()],int(match[1]),tzinfo=timezone.utc)
    except (ValueError, KeyError): return None


def extract_document(url: str, *, fetcher=None) -> NewsDocument:
    try:
        page = (fetcher or fetch_public)(url)
        data = trafilatura.extract(page.content, url=page.url, include_comments=False,
            include_tables=True, output_format="json", with_metadata=True,
            date_extraction_params={"extensive_search":False,"original_date":True})
        if not data:
            raise ExtractionError("Não foi possível extrair o texto principal")
        result = json.loads(data)
        tree = html.fromstring(page.content)
        publication = tree.xpath('//meta[@property="article:published_time"]/@content')
        modification = tree.xpath('//meta[@property="article:modified_time" or @property="og:updated_time"]/@content')
        visible_publication = visible_date(tree,"Publicado")
        visible_modification = visible_date(tree,"Atualizado")
        published_at = _date(publication[0]) if publication else visible_publication or _date(result.get("date"))
        updated_at = _date(modification[0]) if modification else visible_modification
        body = sanitize_text(result.get("text") or "")
        if not body or len(body)>100_000:
            raise ExtractionError("Texto vazio ou acima do limite")
        return NewsDocument(title=(result.get("title") or "")[:500], body=body,
            source_url=page.url, published_at=published_at, updated_at=updated_at,
            date_provenance={"published_at":"article:published_time" if publication else "visible Publicado label; date precision; time/zone unavailable" if visible_publication else "trafilatura original-date extraction (ambiguous)",
                             "updated_at":"article:modified_time/og:updated_time" if modification else "visible Atualizado label; date precision; time/zone unavailable" if visible_modification else "not available",
                             "event_date":"not inferred from publication date"})
    except (FetchError, ValueError) as exc:
        raise ExtractionError(str(exc)) from exc


def extract_news(url: str) -> dict:
    """Compatibility adapter; new routes consume NewsDocument directly."""
    doc = extract_document(url)
    return {"title":doc.title, "author":"", "clean_content":doc.body,
        "published_at":doc.published_at, "source_url":str(doc.source_url),
        "content_hash":doc.content_hash()}
