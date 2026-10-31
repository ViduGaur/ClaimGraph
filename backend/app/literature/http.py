"""Polite HTTP client shared by all literature sources.

* per-host minimum interval between requests (arXiv asks for 1 request / 3 s)
* retries with exponential backoff on 429/5xx, honouring ``Retry-After``
* identifies ClaimGraph with a descriptive User-Agent (+ contact email if set)
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# minimum seconds between requests per host
HOST_INTERVALS = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "api.semanticscholar.org": 1.1,
    "api.openalex.org": 0.15,
    "www.ebi.ac.uk": 0.2,
    "api.unpaywall.org": 0.2,
}


class SourceError(RuntimeError):
    """A literature source failed in a way the caller should report, not crash on."""


class RateLimited(SourceError):
    pass


class PoliteClient:
    def __init__(self, contact_email: str = "", timeout_s: float = 30.0, max_retries: int = 3,
                 transport: httpx.BaseTransport | None = None):
        ua = "ClaimGraph/0.1 (academic claim analysis; student project"
        ua += f"; mailto:{contact_email})" if contact_email else ")"
        self._client = httpx.Client(
            headers={"User-Agent": ua},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self.max_retries = max_retries
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.sleep = time.sleep  # swapped out in tests

    def _wait_turn(self, host: str) -> None:
        interval = HOST_INTERVALS.get(host, 0.0)
        with self._lock:
            now = time.monotonic()
            wait = self._last.get(host, 0.0) + interval - now
            self._last[host] = max(now, now + wait)
        if wait > 0:
            self.sleep(wait)

    def get(self, url: str, **kwargs) -> httpx.Response:
        host = urlparse(url).netloc
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_turn(host)
            try:
                resp = self._client.get(url, **kwargs)
            except httpx.TimeoutException as exc:
                last_exc = exc
                log.warning("timeout on %s (attempt %d)", host, attempt + 1)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning("network error on %s: %s", host, exc)
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    if attempt == self.max_retries or wait > 30:
                        if resp.status_code == 429:
                            raise RateLimited(f"{host} rate limited the request (HTTP 429)")
                        raise SourceError(f"{host} returned HTTP {resp.status_code}")
                    log.info("%s HTTP %d, retrying in %.1fs", host, resp.status_code, wait)
                    self.sleep(wait)
                    delay *= 2
                    continue
                return resp
            if attempt < self.max_retries:
                self.sleep(delay)
                delay *= 2
        raise SourceError(f"{host} unreachable: {last_exc}")

    def stream_bytes(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        """Download up to ``max_bytes``; returns (content, content_type)."""
        host = urlparse(url).netloc
        self._wait_turn(host)
        with self._client.stream("GET", url) as resp:
            if resp.status_code >= 400:
                raise SourceError(f"download failed: HTTP {resp.status_code} from {host}")
            declared = int(resp.headers.get("Content-Length") or 0)
            if declared and declared > max_bytes:
                raise SourceError(f"file too large ({declared // 1_000_000} MB)")
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise SourceError("file exceeded size limit while downloading")
                chunks.append(chunk)
            return b"".join(chunks), resp.headers.get("Content-Type", "")

    def close(self) -> None:
        self._client.close()
