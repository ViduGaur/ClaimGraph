"""Disk cache for LLM calls.

Wrapping a provider in :class:`CachedProvider` makes every (model, prompt, schema) → response
mapping durable: re-running an analysis or an experiment replays the stored responses instead
of calling the model again. With temperature 0 this is what makes LLM-assisted runs reproducible
and lets experiments be re-scored offline. The cache is an append-only JSONL file, so it can be
versioned next to the results it produced.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from app.providers.llm.base import LLMProvider


class CachedProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, path: Path):
        super().__init__(inner.model, inner.timeout_s)
        self.inner = inner
        self.name = inner.name
        self.path = path
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self._store[row["key"]] = row["response"]
                except (ValueError, KeyError):
                    continue  # a torn last line after an interrupted run is not fatal

    def _key(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> str:
        blob = json.dumps([self.inner.name, self.model, getattr(self.inner, "cache_tag", ""), system, user, schema, max_tokens],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def is_available(self) -> bool:
        return self.inner.is_available()

    def complete_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 400) -> dict[str, Any]:
        key = self._key(system, user, schema, max_tokens)
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
        response = self.inner.complete_json(system, user, schema, max_tokens)  # failures propagate and are not cached
        with self._lock:
            self.misses += 1
            self._store[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"key": key, "model": self.model, "response": response}, ensure_ascii=False) + "\n")
        return response
