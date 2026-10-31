"""LLM provider interface.

Nothing outside ``app/providers/llm`` knows which vendor or runtime serves the
model. Pipeline code only calls :meth:`LLMProvider.complete_json`, and every
call goes through :class:`BudgetedLLM`, which enforces a per-analysis call
budget and turns failures into ``None`` so one bad call never breaks a run.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised by providers for any failure (network, timeout, bad output)."""


class LLMProvider(ABC):
    name: str = "abstract"

    def __init__(self, model: str, timeout_s: float = 120.0):
        self.model = model
        self.timeout_s = timeout_s

    @abstractmethod
    def complete_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 400) -> dict[str, Any]:
        """Return a JSON object that should conform to ``schema``."""

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap reachability check (no generation)."""


def _python_literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Leniently parse the first JSON object in a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMError(f"model did not return JSON: {text[:120]!r}")
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            obj = _python_literal(match.group(0))  # some models answer with a Python-style dict (single quotes, None)
            if obj is None:
                raise LLMError(f"unparseable JSON from model: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("model returned JSON that is not an object")
    return obj


class BudgetedLLM:
    """Wraps a provider with a hard call budget and failure isolation."""

    def __init__(self, provider: LLMProvider | None, max_calls: int):
        self.provider = provider
        self.max_calls = max_calls
        self.calls = 0
        self.failures: list[str] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.provider is not None and self.max_calls > 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls) if self.enabled else 0

    def complete_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 400) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            if self.calls >= self.max_calls:
                return None
            self.calls += 1
        try:
            return self.provider.complete_json(system, user, schema, max_tokens)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — provider failures must never escape
            msg = f"{self.provider.name} call failed: {exc}"  # type: ignore[union-attr]
            log.warning(msg)
            self.failures.append(msg)
            return None
