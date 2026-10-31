"""Application settings, read from ``CLAIMGRAPH_*`` environment variables or the repository-root ``.env``.

The prefix keeps generic names such as ``DATABASE_URL``, exported for other projects, from being picked up.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"


DEFAULT_LLM_MODELS = {"openrouter": "inclusionai/ling-3.0-flash", "ollama": "qwen2.5:7b"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAIMGRAPH_",
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- runtime -------------------------------------------------------------
    demo_mode: bool = False
    database_url: str = f"sqlite:///{DATA_DIR / 'claimgraph.db'}"
    data_dir: Path = DATA_DIR
    cors_origins: str = "http://localhost:5199,http://127.0.0.1:5199"
    log_level: str = "INFO"

    # --- literature discovery -----------------------------------------------
    contact_email: str = ""  # sent to OpenAlex / Unpaywall / arXiv as polite contact
    openalex_api_key: str = ""  # optional: free key raises the daily budget 10x
    semantic_scholar_api_key: str = ""  # optional: keyless pool is shared + throttled
    sources: str = "openalex,semantic_scholar,arxiv"  # also: europepmc
    max_candidates_per_source: int = 25
    max_papers: int = 8
    download_pdfs: bool = True
    max_pdf_mb: int = 25
    http_timeout_s: float = 30.0

    # --- local models ---------------------------------------------------------
    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    nli_backend: Literal["onnx", "lexical"] = "onnx"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    model_cache_dir: Path = DATA_DIR / "models"

    # --- LLM (optional) -------------------------------------------------------
    # auto = OpenRouter when CLAIMGRAPH_LLM_API_KEY is set, otherwise no LLM
    llm_provider: Literal["auto", "none", "openrouter", "ollama", "openai_compatible", "huggingface"] = "auto"
    llm_model: str = ""  # empty = the provider's default (see DEFAULT_LLM_MODELS)
    ollama_base_url: str = "http://localhost:11434"
    llm_api_base_url: str = ""  # for openai_compatible (Groq, vLLM, LM Studio...); optional override for openrouter
    llm_api_key: str = ""  # for openrouter / openai_compatible / huggingface
    llm_reasoning: bool = False  # openrouter: let reasoning models think before answering (slower, less reproducible)
    llm_route: str = "Novita"  # openrouter: preferred upstream host for the model ("" = any)
    llm_route_strict: bool = False  # openrouter: never fall back to another host (set for experiments)
    llm_timeout_s: float = 180.0
    llm_max_calls: int = Field(0, description="hard budget of LLM calls per analysis; 0 = provider default")
    llm_for_extraction: bool = False  # LLM writes the claims itself (grounding-checked); off = LLM only selects
    llm_claim_selection: bool = True  # with an LLM: it screens the verbatim cue-based candidates (1 call per paper)
    llm_pair_analysis: bool = True  # with an LLM: grounded stance + context comparison per candidate pair
    llm_conflict_explanations: bool = True  # with an LLM: grounded wording of each conflict verdict (after the decision)
    llm_explanation_budget: int = 12  # LLM calls kept back for those explanations
    llm_concurrency: int = 6  # parallel LLM calls (hosted providers; keep 1 for a local CPU model)
    llm_cache: bool = True  # replay identical LLM calls from data_dir/llm_cache.jsonl

    # --- reasoning thresholds -------------------------------------------------
    max_claims_per_paper: int = 6
    pair_similarity_threshold: float = 0.0  # 0 = auto (depends on embedding backend)
    max_pairs: int = 120

    @property
    def llm_provider_name(self) -> str:
        """The provider actually used, with ``auto`` resolved."""
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openrouter" if self.llm_api_key else "none"

    @property
    def llm_call_budget(self) -> int:
        """A local CPU model takes about a minute per call; a hosted one a few seconds."""
        if self.llm_max_calls > 0:
            return self.llm_max_calls
        return 6 if self.llm_provider_name == "ollama" else 80

    @property
    def llm_model_name(self) -> str:
        return self.llm_model or DEFAULT_LLM_MODELS.get(self.llm_provider_name, "")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def source_list(self) -> list[str]:
        return [s.strip().lower() for s in self.sources.split(",") if s.strip()]

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"


@lru_cache
def get_settings() -> Settings:
    return Settings()
