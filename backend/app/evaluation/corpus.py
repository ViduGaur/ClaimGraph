"""Frozen topic corpora.

A corpus is built once (seed arXiv papers + automatic discovery, full text where an
open-access copy exists) and stored as a JSON snapshot. Every later stage reads the
snapshot, so experiments are offline, deterministic and independent of API drift.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.evaluation import CORPORA_DIR, TOPICS_PATH
from app.extraction.acquisition import Acquirer, PaperText
from app.extraction.text import Section
from app.literature.discovery import deduplicate, discover, rank_papers
from app.literature.http import PoliteClient, SourceError
from app.literature.sources import ArxivSource, OpenAlexSource, SemanticScholarSource
from app.providers.llm import BudgetedLLM
from app.providers.models import ModelBundle
from app.reasoning.query import understand_query
from app.schemas.domain import Paper

log = logging.getLogger(__name__)


@dataclass
class Topic:
    id: str
    split: str  # "dev" | "test"
    question: str
    seeds: list[str]


def load_topics() -> tuple[list[Topic], int]:
    raw = json.loads(TOPICS_PATH.read_text())
    return [Topic(t["id"], t["split"], t["question"], t.get("seeds", [])) for t in raw["topics"]], int(raw.get("n_papers", 10))


def get_topic(topic_id: str) -> Topic:
    for t in load_topics()[0]:
        if t.id == topic_id:
            return t
    raise KeyError(f"unknown topic {topic_id!r} (see {TOPICS_PATH})")


def corpus_path(topic_id: str):
    return CORPORA_DIR / f"{topic_id}.json"


def build_corpus(topic: Topic, n_papers: int, settings: Settings, models: ModelBundle, *, discover_extra: bool = True,
                 min_relevance: float = 0.5) -> dict:
    """Seeds first, then the best automatically discovered papers, up to ``n_papers``."""
    client = PoliteClient(settings.contact_email, settings.http_timeout_s)
    notes: list[str] = []
    try:
        arxiv = ArxivSource(client)
        seeds = arxiv.fetch_ids(topic.seeds)
        missing = set(topic.seeds) - {p.source_ids["arxiv"] for p in seeds}
        if missing:
            notes.append(f"seed ids not returned by arXiv: {sorted(missing)}")
        selected: list[Paper] = [p.model_copy(update={"sources": ["seed"]}) for p in seeds]
        if discover_extra and len(selected) < n_papers:
            plan = understand_query(topic.question, BudgetedLLM(None, 0))
            sources = [OpenAlexSource(client, settings.openalex_api_key, settings.contact_email),
                       SemanticScholarSource(client, settings.semantic_scholar_api_key), arxiv]
            found = discover(plan.search_queries, sources, settings.max_candidates_per_source)
            notes.extend(found.warnings)
            ranked = rank_papers(topic.question, found.candidates, models.reranker)
            analysable = [p for p in ranked if p.abstract and (p.pdf_url or p.source_ids.get("arxiv"))
                          and p.relevance >= min_relevance]
            for p in analysable:
                if len(selected) >= n_papers:
                    break
                if len(deduplicate([*selected, p])) > len(selected):
                    selected.append(p)
        acquirer = Acquirer(client, settings.papers_dir, settings.contact_email, settings.max_pdf_mb, True)
        docs: list[PaperText] = []
        for p in selected[:n_papers]:
            try:
                docs.append(acquirer.acquire(p))
            except (SourceError, ValueError) as exc:
                notes.append(f"{p.citation}: acquisition failed ({exc})")
                if p.abstract:
                    docs.append(PaperText(p.model_copy(update={"text_source": "abstract"}), [Section("abstract", p.abstract)]))
    finally:
        client.close()
    snapshot = {
        "topic": topic.__dict__,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": notes,
        "papers": [{"paper": d.paper.model_dump(), "sections": [{"name": s.name, "text": s.text} for s in d.sections]}
                   for d in docs if d.sections],
    }
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path(topic.id).write_text(json.dumps(snapshot, ensure_ascii=False, indent=1))
    return snapshot


def load_corpus(topic_id: str) -> list[PaperText]:
    path = corpus_path(topic_id)
    if not path.exists():
        raise FileNotFoundError(f"no corpus snapshot for {topic_id!r}; run: scripts/cg_eval.py build-corpus --topic {topic_id}")
    data = json.loads(path.read_text())
    return [PaperText(Paper(**item["paper"]), [Section(s["name"], s["text"]) for s in item["sections"]]) for item in data["papers"]]


def describe_corpus(topic_id: str) -> str:
    docs = load_corpus(topic_id)
    lines = [f"{topic_id}: {len(docs)} papers"]
    for d in docs:
        chars = sum(len(s.text) for s in d.sections)
        origin = "seed" if "seed" in d.paper.sources else "discovered"
        lines.append(f"  {d.paper.paper_id:22} {d.paper.text_source:9} {chars:7d} chars  {origin:10} {d.paper.citation}: {d.paper.title[:70]}")
    return "\n".join(lines)
