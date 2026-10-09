import json
import pytest

from ml.contracts import Claim
from scraper.providers import FallbackProvider, SearxngProvider, configured_provider
from scraper.safe_http import FetchedPage

def test_fallback_external_service(monkeypatch):
    monkeypatch.delenv("SEARXNG_BASE_URL",raising=False)
    assert isinstance(configured_provider(),FallbackProvider)


def test_search_links_must_be_safe_and_are_not_evidence():
    body=json.dumps({"results":[{"url":"https://127.0.0.1/x"},{"url":"https://a.example/x"},{"url":"https://a.example/x"},{"url":None}]})
    provider=SearxngProvider("https://search.example",fetcher=lambda u,**k:FetchedPage(u,body.encode(),"application/json"))
    assert list(provider.search(Claim(id="a",text="O município registrou 82 casos.")).keys())==["https://a.example/x"]


@pytest.mark.parametrize("body", ['[]','{"results": null}', '{"results":"unexpected"}'])
def test_provider_contract_failure_is_controlled(body):
    provider=SearxngProvider("https://search.example",fetcher=lambda u,**k:FetchedPage(u,body.encode(),"application/json"))
    assert provider.search(Claim(id="a",text="Alegação de teste.")) == {}

