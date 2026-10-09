"""Source identity helpers. Categories describe a publisher; they are never truth labels."""
from urllib.parse import urlsplit

from tld import get_fld

# Brazilian public-body second-level domains (registro.br categories).
PUBLIC_BODY_SUFFIXES = (".gov.br", ".leg.br", ".jus.br", ".mp.br", ".def.br", ".mil.br")
RESEARCH_SUFFIXES = (".edu.br",)


def hostname(url) -> str:
    return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")


def registrable_domain(url) -> str:
    """Publisher grouping: subdomains of one registrable domain are one origin."""
    if not url:
        return ""
    host = hostname(url)
    return get_fld(str(url), fail_silently=True) or host


def source_category(url) -> str:
    host = "." + hostname(url)
    if host.endswith(PUBLIC_BODY_SUFFIXES):
        return "orgao_publico_br"
    if host.endswith(RESEARCH_SUFFIXES):
        return "ensino_pesquisa_br"
    return "nao_classificada"
