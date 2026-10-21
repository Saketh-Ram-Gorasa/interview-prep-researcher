"""Public web discovery (ddgs) and page fetching under the config source policy."""

import ipaddress
import socket
from functools import cache
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from ddgs import DDGS

from backend import config
from backend.schemas import Tier


def search(query: str, max_results: int) -> list[dict]:
    """Search results as ddgs returns them ({title, href, body}), minus blocked domains."""
    hits = DDGS().text(query, region=config.SEARCH_REGION, max_results=max_results)
    return [h for h in hits if not _on_domains(h["href"], config.BLOCKED_DOMAINS)]


def fetch(url: str) -> str:
    """Return a public page's HTML. Raises if the URL, a redirect, or robots.txt refuses it."""
    if not _robots(_origin(url)).can_fetch(config.USER_AGENT, url):
        raise PermissionError("robots.txt disallows this page")
    with _client() as client:
        return client.get(url).raise_for_status().text


def tier(url: str, company: str) -> Tier:
    """Credibility tier: the company's own domains, curated community sites, or other."""
    if _on_domains(url, config.COMPANIES.get(company, {}).get("domains", [])):
        return "official"
    if _on_domains(url, config.COMMUNITY_DOMAINS):
        return "community"
    return "other"


def _client() -> httpx.Client:
    """HTTP client that checks every request, including each redirect hop."""
    return httpx.Client(
        follow_redirects=True,
        timeout=config.FETCH_TIMEOUT_S,
        headers={"User-Agent": config.USER_AGENT},
        event_hooks={"request": [lambda request: _check_allowed(str(request.url))]},
    )


def _check_allowed(url: str) -> None:
    """Only public http(s) hosts outside the blocked list may be fetched."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise PermissionError(f"not an http(s) URL: {url}")
    if _on_domains(url, config.BLOCKED_DOMAINS):
        raise PermissionError(f"blocked domain: {parts.hostname}")
    for *_, address in socket.getaddrinfo(parts.hostname, None):
        if not ipaddress.ip_address(address[0]).is_global:
            raise PermissionError(f"not a public address: {parts.hostname}")


@cache
def _robots(origin: str) -> RobotFileParser:
    """robots.txt rules per site: 401/403 disallows everything, a missing file allows everything."""
    rules = RobotFileParser()
    with _client() as client:
        response = client.get(f"{origin}/robots.txt")
    if response.status_code in (401, 403):
        rules.disallow_all = True
    elif response.is_success:
        rules.parse(response.text.splitlines())
    else:
        rules.allow_all = True
    return rules


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _on_domains(url: str, domains) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == d or host.endswith(f".{d}") for d in domains)
