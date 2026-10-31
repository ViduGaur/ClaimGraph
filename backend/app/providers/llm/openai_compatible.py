"""Optional hosted LLMs through the OpenAI-compatible chat-completions protocol.

One small client covers many free-tier or self-hosted endpoints (OpenRouter,
Groq, Together, vLLM, LM Studio, llama.cpp server) and the Hugging Face
Inference Providers router. None of them is required to run ClaimGraph.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.providers.llm.base import LLMError, LLMProvider, parse_json_object

HF_ROUTER_URL = "https://router.huggingface.co/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
JSON_REPAIR_NOTE = ("Reminder: answer with one valid JSON object only. Inside string values, escape every double quote "
                    "as \\\" (or use single quotes), and do not truncate the object.")


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    max_retries = 3  # on timeouts, 429 and 5xx, with exponential backoff
    json_mode = True  # send response_format=json_object (not every upstream supports it)

    def __init__(self, model: str, base_url: str, api_key: str, timeout_s: float = 120.0):
        super().__init__(model, timeout_s)
        if not base_url:
            raise LLMError("CLAIMGRAPH_LLM_API_BASE_URL is required for the openai_compatible provider")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _extra_payload(self, max_tokens: int) -> dict[str, Any]:
        """Provider-specific request fields (overrides ``max_tokens`` etc.)."""
        return {}

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            return r.status_code < 400
        except Exception:  # noqa: BLE001
            return False

    def complete_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 400) -> dict[str, Any]:
        schema_hint = ("\n\nRespond with ONLY a valid JSON object (double-quoted keys and strings, no markdown, no comments) "
                       f"matching this JSON schema:\n{json.dumps(schema)}")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            **({"response_format": {"type": "json_object"}} if self.json_mode else {}),
            "messages": [
                {"role": "system", "content": system + schema_hint},
                {"role": "user", "content": user},
            ],
            **self._extra_payload(max_tokens),
        }
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers(), timeout=self.timeout_s)
            except httpx.TimeoutException as exc:
                if last:
                    raise LLMError(f"{self.name} timed out") from exc
            except httpx.HTTPError as exc:
                if last:
                    raise LLMError(f"{self.name} unreachable: {exc}") from exc
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    if last:
                        raise LLMError(f"{self.name} rate limited (HTTP 429)" if r.status_code == 429
                                       else f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
                elif r.status_code >= 400:
                    raise LLMError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}")
                else:
                    try:
                        content = r.json()["choices"][0]["message"]["content"]
                        return parse_json_object(content or "")
                    except (KeyError, IndexError, TypeError, ValueError, LLMError) as exc:
                        # hosted routers occasionally return an empty or truncated body; retry before giving up
                        if last:
                            raise LLMError(f"{self.name} returned an unexpected payload: {exc}") from exc
                        if len(payload["messages"]) == 2:
                            # at temperature 0 the same prompt yields the same malformed answer, so say what was wrong
                            payload["messages"].append({"role": "user", "content": JSON_REPAIR_NOTE})
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise LLMError(f"{self.name} failed after {self.max_retries + 1} attempts")


class HuggingFaceProvider(OpenAICompatibleProvider):
    """Hugging Face Inference Providers (needs a free HF token; small monthly credit)."""

    name = "huggingface"

    def __init__(self, model: str, api_key: str, timeout_s: float = 120.0):
        if not api_key:
            raise LLMError("CLAIMGRAPH_LLM_API_KEY (a Hugging Face token) is required for the huggingface provider")
        super().__init__(model, HF_ROUTER_URL, api_key, timeout_s)


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter (https://openrouter.ai): one key, many hosted models.

    Requests are made deterministic where the API allows it: temperature 0, a fixed seed, and —
    by default — reasoning switched off, because a reasoning model otherwise spends the output
    budget on hidden thinking and returns an empty answer. Together with the on-disk response
    cache this is what makes LLM-assisted runs repeatable.
    """

    name = "openrouter"
    REASONING_HEADROOM = 3000  # extra output tokens for the hidden reasoning when it is enabled
    # OpenRouter serves one model through several upstream hosts, which differ in speed, quantisation and
    # supported parameters. Requiring response_format restricts routing to the hosts that implement it,
    # so JSON is requested in the prompt instead and parsed leniently.
    json_mode = False

    def __init__(self, model: str, api_key: str, timeout_s: float = 120.0, base_url: str = "", reasoning: bool = False,
                 seed: int = 7, route: str = "", strict_route: bool = False):
        if not api_key:
            raise LLMError("CLAIMGRAPH_LLM_API_KEY (an OpenRouter key) is required for the openrouter provider")
        super().__init__(model, base_url or OPENROUTER_URL, api_key, timeout_s)
        self.reasoning = reasoning
        self.seed = seed
        self.route = route  # preferred upstream host, e.g. "Novita"; "" = let OpenRouter choose
        self.strict_route = strict_route  # True = never fall back to another host (experiments)

    @property
    def cache_tag(self) -> str:
        """Distinguishes cached responses produced under different routing/reasoning settings."""
        return f"route={self.route or 'any'}{'!' if self.strict_route else ''};reasoning={int(self.reasoning)};seed={self.seed}"

    def _headers(self) -> dict[str, str]:
        return {**super()._headers(), "X-Title": "ClaimGraph"}

    def _extra_payload(self, max_tokens: int) -> dict[str, Any]:
        extra: dict[str, Any] = {"seed": self.seed, "reasoning": {"enabled": self.reasoning}}
        if self.reasoning:
            extra["max_tokens"] = max_tokens + self.REASONING_HEADROOM
        if self.route:
            extra["provider"] = {"order": [self.route], "allow_fallbacks": not self.strict_route}
        return extra

    def is_available(self) -> bool:
        return True  # the public /models endpoint says nothing about this key; real failures surface per call
