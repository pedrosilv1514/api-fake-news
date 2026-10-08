"""Bounded HTTPS only, with DNS validation and connection to the validated IP."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import http.client
import ipaddress
import socket
import ssl
import time
import threading
import certifi
from urllib.parse import urljoin, urlsplit

MAX_BYTES = 2_000_000
DNS_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="public-dns")
DNS_SLOTS = threading.BoundedSemaphore(4)


class FetchError(Exception):
    pass


def validate_url(url: str):
    if len(url)>2048:
        raise FetchError("URL excede 2048 caracteres")
    if any(ord(c) < 32 or c.isspace() for c in url):
        raise FetchError("URL contém caracteres inválidos")
    try:
        p = urlsplit(url)
        port = p.port
    except ValueError as exc:
        raise FetchError("URL inválida") from exc
    if p.scheme != "https" or not p.hostname or p.username or p.password or port not in (None,443):
        raise FetchError("Use URL HTTPS pública, sem credenciais ou porta alternativa")
    try:
        hostname = p.hostname.encode("idna").decode().lower()
    except UnicodeError as exc:
        raise FetchError("Hostname inválido") from exc
    if hostname == "localhost" or hostname.endswith((".localhost",".local")):
        raise FetchError("Endereço local não permitido")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise FetchError("IP não público")
    return p, hostname


def resolve_public(hostname: str, timeout=3):
    if not DNS_SLOTS.acquire(blocking=False):
        raise FetchError("Resolvedor DNS ocupado")
    try:
        future = DNS_POOL.submit(socket.getaddrinfo,hostname,443,type=socket.SOCK_STREAM)
    except Exception:
        DNS_SLOTS.release()
        raise
    future.add_done_callback(lambda task: DNS_SLOTS.release())
    try:
        infos = future.result(timeout=timeout)
        addresses = list(dict.fromkeys(info[4][0] for info in infos))
    except FutureTimeout as exc:
        raise FetchError("Prazo de DNS excedido") from exc
    except OSError as exc:
        raise FetchError("Falha de DNS") from exc
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise FetchError("DNS aponta para rede não pública")
    return addresses


@dataclass(frozen=True)
class FetchedPage:
    url: str
    content: bytes
    content_type: str


def fetch_public(url: str, timeout: float = 10, max_bytes: int = MAX_BYTES):
    deadline = time.monotonic() + timeout
    for _ in range(4):
        parsed, hostname = validate_url(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise FetchError("Prazo de download excedido")
        addresses = resolve_public(hostname,timeout=min(3,remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FetchError("Prazo de download excedido")
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
        conn = http.client.HTTPSConnection(hostname,443,timeout=remaining,context=context)
        # HTTPS still uses the original hostname for SNI/certificate verification.
        # The socket connects to the vetted IP, with no second DNS lookup of hostname.
        conn._create_connection = lambda address, timeout=None, source_address=None: socket.create_connection((addresses[0],443),timeout,source_address)
        path = (parsed.path or "/") + ("?"+parsed.query if parsed.query else "")
        try:
            conn.request("GET",path,headers={"User-Agent":"FakeNewsResearch/0.2","Accept-Encoding":"identity"})
            response = conn.getresponse()
            if response.status in (301,302,303,307,308):
                location = response.getheader("Location")
                if not location:
                    raise FetchError("Redirecionamento sem destino")
                url = urljoin(url,location)
                validate_url(url)
                continue
            if response.status != 200:
                raise FetchError(f"Fonte retornou HTTP {response.status}")
            if response.getheader("Content-Encoding","identity").lower() not in ("identity",""):
                raise FetchError("Conteúdo comprimido não suportado")
            content_type = response.getheader("Content-Type","").split(";")[0].lower()
            if content_type not in ("text/html","application/xhtml+xml","application/json","application/rss+xml","application/xml","text/xml"):
                raise FetchError("Tipo de conteúdo não permitido")
            length = response.getheader("Content-Length")
            if length and (not length.isdigit() or int(length)>max_bytes):
                raise FetchError("Resposta excede limite de tamanho")
            chunks, total = [], 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise FetchError("Prazo de download excedido")
                if conn.sock is not None: conn.sock.settimeout(remaining)
                chunk = response.read(min(65_536,max_bytes+1-total))
                if not chunk: break
                chunks.append(chunk); total += len(chunk)
                if total>max_bytes: raise FetchError("Resposta excede limite de tamanho")
            return FetchedPage(url,b"".join(chunks),content_type)
        except (OSError, http.client.HTTPException) as exc:
            raise FetchError("Não foi possível baixar a fonte") from exc
        finally:
            conn.close()
    raise FetchError("Redirecionamentos demais")
