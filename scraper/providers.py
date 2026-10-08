"""Evidence discovery is separate from extraction and judgment."""
import json
import os
from typing import Protocol
from urllib.parse import urlencode

from ml.contracts import Claim
from scraper.safe_http import FetchError, fetch_public, validate_url


class SearchProvider(Protocol):
    def search(self, claim: Claim) -> list[str]: ...


class NoSearchProvider:
    def search(self, claim):
        return []


class SearxngProvider:
    """Operator-owned HTTPS SearxNG; no default public instance or paid calls."""
    def __init__(self, base_url, fetcher=fetch_public):
        validate_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.fetcher = fetcher

    def search(self, claim):
        url = self.base_url + "/search?" + urlencode({"q":claim.text[:300],"format":"json","language":"pt-BR"})
        page = self.fetcher(url, timeout=8, max_bytes=300_000)
        data = json.loads(page.content)
        if not isinstance(data,dict) or not isinstance(data.get("results", []),list):
            raise FetchError("Resposta de busca fora do contrato")
        urls = []
        for row in data.get("results", [])[:20]:
            if not isinstance(row,dict): continue
            candidate = row.get("url", "")
            if not isinstance(candidate,str): continue
            try: validate_url(candidate)
            except FetchError: continue
            if candidate not in urls: urls.append(candidate)
            if len(urls)==3: break
        return urls


def configured_provider():
    endpoint = os.getenv("SEARXNG_BASE_URL")
    return SearxngProvider(endpoint) if endpoint else NoSearchProvider()
