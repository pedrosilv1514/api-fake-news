import socket
from unittest.mock import Mock

import pytest

from scraper.safe_http import FetchError, fetch_public, resolve_public, validate_url


@pytest.mark.parametrize("url",[
    "http://example.com", "https://localhost/a", "https://127.0.0.1/a",
    "https://10.0.0.1", "https://169.254.169.254", "https://[::1]",
    "https://user:password@example.com", "https://example.com:8080",
    "https://example.local", "file:///etc/passwd", "https://example.com/a\nb",
])
def test_disallowed_urls(url):
    with pytest.raises(FetchError): validate_url(url)


def test_mixed_public_private_dns_rejected(monkeypatch):
    monkeypatch.setattr(socket,"getaddrinfo",lambda *a,**k:[
        (socket.AF_INET,socket.SOCK_STREAM,6,"",("93.184.216.34",443)),
        (socket.AF_INET,socket.SOCK_STREAM,6,"",("127.0.0.1",443))])
    with pytest.raises(FetchError,match="não pública"): resolve_public("example.com")


def mock_connection(monkeypatch, response):
    import scraper.safe_http as safe
    monkeypatch.setattr(safe,"resolve_public",lambda h,**kwargs:["93.184.216.34"])
    connection = Mock()
    connection.getresponse.return_value = response
    connection.sock = None
    monkeypatch.setattr(safe.http.client,"HTTPSConnection",lambda *a,**k: connection)
    return connection


def test_redirect_to_private_rejected_before_second_connection(monkeypatch):
    response = Mock(status=302)
    response.getheader.return_value = "https://127.0.0.1/"
    conn = mock_connection(monkeypatch,response)
    with pytest.raises(FetchError): fetch_public("https://example.com")
    assert conn.request.call_count==1
    conn.close.assert_called_once()


def test_large_response_and_compression_rejected(monkeypatch):
    response = Mock(status=200)
    headers = {"Content-Encoding":"identity","Content-Type":"text/html","Content-Length":"2000001"}
    response.getheader.side_effect = lambda k,default=None: headers.get(k,default)
    mock_connection(monkeypatch,response)
    with pytest.raises(FetchError,match="tamanho"): fetch_public("https://example.com")
    headers["Content-Encoding"]="gzip"
    with pytest.raises(FetchError,match="comprimido"): fetch_public("https://example.com")


def test_dns_pinned_socket_uses_vetted_ip(monkeypatch):
    response = Mock(status=200)
    headers = {"Content-Type":"text/html","Content-Encoding":"identity"}
    response.getheader.side_effect = lambda k,default=None: headers.get(k,default)
    response.read.side_effect = [b"<html>source</html>", b""]
    conn = mock_connection(monkeypatch,response)
    connect = Mock()
    monkeypatch.setattr(socket,"create_connection",connect)
    page = fetch_public("https://example.com/a")
    conn._create_connection(("example.com",443),timeout=2)
    assert connect.call_args.args[0] == ("93.184.216.34",443)
    assert page.content == b"<html>source</html>"
