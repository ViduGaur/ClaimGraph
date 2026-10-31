"""Shared fixtures. Tests never hit the network and never download models:
they use the hashing embedder + lexical NLI fallbacks unless marked `models`."""
from __future__ import annotations

import os

import pytest

from app.core.config import Settings, get_settings
from app.providers.llm import BudgetedLLM
from app.providers.models import EmbeddingReranker, HashingEmbedder, LexicalNLI, ModelBundle, NLIModel
from app.schemas.domain import Claim, ClaimContext, ClaimType, Paper, Passage, Polarity


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        demo_mode=True,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        data_dir=tmp_path,
        embedding_backend="hashing",
        nli_backend="lexical",
        llm_provider="none",
        _env_file=None,
    )


@pytest.fixture
def models() -> ModelBundle:
    emb = HashingEmbedder()
    return ModelBundle(emb, EmbeddingReranker(emb), LexicalNLI(), [])


@pytest.fixture
def no_llm() -> BudgetedLLM:
    return BudgetedLLM(None, 0)


class FixedNLI(NLIModel):
    """Returns the same probabilities for every pair — lets tests pin the NLI signal."""

    name = "fixed"

    def __init__(self, entailment: float, contradiction: float):
        self.e, self.c = entailment, contradiction

    def predict(self, pairs):
        return [{"entailment": self.e, "contradiction": self.c, "neutral": max(0.0, 1 - self.e - self.c)} for _ in pairs]


@pytest.fixture
def fixed_nli():
    return FixedNLI


def make_paper(pid: str, year: int = 2021, **kw) -> Paper:
    return Paper(paper_id=pid, title=f"Paper {pid}", authors=[f"Author {pid}", "Second Author"], year=year,
                 abstract="An abstract.", **kw)


def make_claim(cid: str, pid: str, text: str, *, subject: str | None = "ViT", polarity: Polarity = Polarity.POSITIVE,
               ctype: ClaimType = ClaimType.COMPARISON, confidence: float = 0.8, **ctx) -> Claim:
    return Claim(
        claim_id=cid, paper_id=pid, claim_text=text, normalized_text=text, claim_type=ctype, polarity=polarity,
        subject=subject, evidence=[Passage(passage_id=f"p_{cid}", paper_id=pid, section="abstract", text=text)],
        section="abstract", context=ClaimContext(**ctx), confidence=confidence, source_citation=f"Author {pid} et al. (2021)",
    )


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """Environment for spinning up the real FastAPI app against a temp DB with light models."""
    monkeypatch.setenv("CLAIMGRAPH_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("CLAIMGRAPH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CLAIMGRAPH_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("CLAIMGRAPH_NLI_BACKEND", "lexical")
    monkeypatch.setenv("CLAIMGRAPH_LLM_PROVIDER", "none")
    monkeypatch.setenv("CLAIMGRAPH_DEMO_MODE", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


os.environ.setdefault("HF_HUB_OFFLINE", "1")
