"""Polite HTTP with an on-disk cache.

FIS pages are static once a race is over, so caching keeps re-scrapes cheap and
keeps us from hammering their servers while iterating on parsers.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

USER_AGENT = (
    "xcpredict/0.1 (+https://github.com/deknapp/xcpredict) "
    "personal cross-country result analysis"
)
DEFAULT_CACHE = Path("data/cache")


class Fetcher:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE, delay_s: float = 1.0,
                 timeout_s: float = 30.0, use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.use_cache = use_cache
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:20]
        return self.cache_dir / f"{digest}.html"

    def get(self, url: str, force: bool = False) -> str:
        path = self._cache_path(url)
        if self.use_cache and not force and path.exists():
            return path.read_text(encoding="utf-8")

        wait = self.delay_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        log.info("GET %s", url)
        response = self._session.get(url, timeout=self.timeout_s)
        self._last_request = time.monotonic()
        response.raise_for_status()

        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
        return response.text

    def get_optional(self, url: str, force: bool = False) -> Optional[str]:
        """Like get(), but returns None on a 404 instead of raising."""
        try:
            return self.get(url, force=force)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
