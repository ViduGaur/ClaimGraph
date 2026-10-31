"""Local LLM via Ollama's HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md)."""
from __future__ import annotations

from typing import Any

import httpx

from app.providers.llm.base import LLMError, LLMProvider, parse_json_object


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout_s: float = 180.0):
        super().__init__(model, timeout_s)
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:  # noqa: BLE001
            return False

    def complete_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 400) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,  # Ollama structured outputs: constrained to the JSON schema
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMError(f"ollama timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable: {exc}") from exc
        if r.status_code == 404:
            raise LLMError(f"model {self.model!r} not found in Ollama (run: ollama pull {self.model})")
        if r.status_code >= 400:
            raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:200]}")
        return parse_json_object(r.json().get("message", {}).get("content", ""))
