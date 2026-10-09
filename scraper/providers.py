"""Evidence discovery is separate from extraction and judgment.

A search provider returns a mapping of URL to a provenance dictionary describing
how and why it was discovered (e.g. search query, rank, provider name).
"""
import json
import os
from typing import Protocol
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

from ml.contracts import Claim
from ml.terms import build_query
from scraper.safe_http import FetchError, fetch_public, validate_url


class SearchProvider(Protocol):
    def search(self, claim: Claim) -> dict[str, dict]: ...


class NoSearchProvider:
    def search(self, claim):
        return {}


class BingNewsRssProvider:
    """Public Bing News RSS interface without API key (rate limited)."""
    def __init__(self, fetcher=fetch_public):
        self.fetcher = fetcher

    def search(self, claim):
        query = build_query(claim.text, claim.context)
        if not query:
            return {}
        url = "https://www.bing.com/news/search?" + urlencode({"q": query, "format": "rss", "setlang": "pt-BR", "cc": "BR"})
        try:
            page = self.fetcher(url, timeout=8, max_bytes=500_000)
            root = ET.fromstring(page.content)
            results = {}
            for rank, item in enumerate(root.iter("item"), 1):
                link = item.findtext("link")
                if not link: continue
                # Extract actual target URL from Bing's tracking link if present
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(link)
                if "apiclick.aspx" in parsed.path:
                    query_params = parse_qs(parsed.query)
                    if "url" in query_params:
                        link = query_params["url"][0]
                # Fallback to https if Bing returned http for some reason
                if link.startswith("http://"):
                    link = "https://" + link[7:]
                try: validate_url(link)
                except FetchError: continue
                if link not in results:
                    results[link] = {
                        "provider": "bing_news_rss",
                        "query": query,
                        "rank": rank,
                        "title": item.findtext("title") or ""
                    }
                if len(results) == 3: break
            return results
        except (FetchError, ET.ParseError):
            return {}


class SearxngProvider:
    """Operator-owned HTTPS SearxNG; requires SEARXNG_BASE_URL."""
    def __init__(self, base_url, fetcher=fetch_public):
        validate_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.fetcher = fetcher

    def search(self, claim):
        query = build_query(claim.text, claim.context)
        if not query:
            return {}
        url = self.base_url + "/search?" + urlencode({"q": query, "format": "json", "language": "pt-BR"})
        try:
            page = self.fetcher(url, timeout=8, max_bytes=300_000)
            data = json.loads(page.content)
            if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
                return {}
            results = {}
            for rank, row in enumerate(data.get("results", []), 1):
                if not isinstance(row, dict): continue
                candidate = row.get("url", "")
                if not isinstance(candidate, str): continue
                try: validate_url(candidate)
                except FetchError: continue
                if candidate not in results:
                    results[candidate] = {
                        "provider": "searxng",
                        "query": query,
                        "rank": rank,
                        "engine": row.get("engine", "")
                    }
                if len(results) == 3: break
            return results
        except (FetchError, json.JSONDecodeError):
            return {}


class FallbackProvider:
    def __init__(self, providers):
        self.providers = providers
    
    def search(self, claim):
        for provider in self.providers:
            results = provider.search(claim)
            if results: return results
        return {}


def configured_provider():
    endpoint = os.getenv("SEARXNG_BASE_URL")
    if endpoint:
        try:
            return SearxngProvider(endpoint)
        except FetchError:
            pass
    return FallbackProvider([BingNewsRssProvider()])
