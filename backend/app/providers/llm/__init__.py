"""LLM provider factory: LLMProvider ├── OllamaProvider ├── HuggingFaceProvider └── OpenAICompatibleProvider."""
from __future__ import annotations

import logging

from app.core.config import Settings
from app.providers.llm.base import BudgetedLLM, LLMError, LLMProvider
from app.providers.llm.cache import CachedProvider
from app.providers.llm.ollama import OllamaProvider
from app.providers.llm.openai_compatible import HuggingFaceProvider, OpenAICompatibleProvider, OpenRouterProvider

log = logging.getLogger(__name__)

__all__ = ["BudgetedLLM", "CachedProvider", "LLMError", "LLMProvider", "build_llm", "create_provider"]


def create_provider(settings: Settings) -> LLMProvider | None:
    kind, model = settings.llm_provider_name, settings.llm_model_name
    if kind == "none":
        return None
    if kind == "openrouter":
        return OpenRouterProvider(model, settings.llm_api_key, settings.llm_timeout_s, settings.llm_api_base_url,
                                  settings.llm_reasoning, route=settings.llm_route, strict_route=settings.llm_route_strict)
    if kind == "ollama":
        return OllamaProvider(model, settings.ollama_base_url, settings.llm_timeout_s)
    if kind == "huggingface":
        return HuggingFaceProvider(model, settings.llm_api_key, settings.llm_timeout_s)
    if kind == "openai_compatible":
        return OpenAICompatibleProvider(model, settings.llm_api_base_url, settings.llm_api_key, settings.llm_timeout_s)
    raise LLMError(f"unknown LLM provider {kind!r}")


def build_llm(settings: Settings, use_llm: bool | None) -> tuple[BudgetedLLM, str | None]:
    """Build the budgeted LLM for one analysis. Returns (llm, warning_or_None)."""
    if use_llm is False or settings.llm_provider_name == "none":
        return BudgetedLLM(None, 0), None
    try:
        provider = create_provider(settings)
    except LLMError as exc:
        return BudgetedLLM(None, 0), f"LLM disabled: {exc}"
    if provider is None or not provider.is_available():
        return BudgetedLLM(None, 0), (
            f"LLM provider '{settings.llm_provider_name}' with model '{settings.llm_model_name}' is not reachable; "
            "continuing with local rule-based reasoning only"
        )
    if settings.llm_cache:
        provider = CachedProvider(provider, settings.data_dir / "llm_cache.jsonl")
    return BudgetedLLM(provider, settings.llm_call_budget), None
