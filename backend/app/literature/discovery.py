"""Literature discovery: multi-source search → deduplication → ranking."""
from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.literature.http import SourceError
from app.literature.sources import LiteratureSource
from app.providers.models import Reranker
from app.schemas.domain import Paper

log = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", title.lower())).strip()


def _merge(into: Paper, other: Paper) -> Paper:
    """Merge metadata of a duplicate record into the canonical one."""
    data = into.model_dump()
    for key in ("year", "venue", "doi", "url", "pdf_url", "oa_status"):
        if not data.get(key) and getattr(other, key):
            data[key] = getattr(other, key)
    if other.abstract and len(other.abstract) > len(data.get("abstract") or ""):
        data["abstract"] = other.abstract
    if len(other.authors) > len(data["authors"]):
        data["authors"] = other.authors
    # arXiv PDFs are the most reliably downloadable open copy
    if other.pdf_url and "arxiv.org" in other.pdf_url and "arxiv.org" not in (data.get("pdf_url") or ""):
        data["pdf_url"] = other.pdf_url
    data["is_open_access"] = into.is_open_access or other.is_open_access
    data["citation_count"] = max(into.citation_count or 0, other.citation_count or 0) or None
    data["sources"] = sorted(set(into.sources) | set(other.sources))
    data["source_ids"] = {**other.source_ids, **into.source_ids}
    return Paper(**data)


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Collapse records describing the same work (DOI, arXiv id, or near-identical title)."""
    canonical: list[Paper] = []
    for paper in papers:
        match_idx = None
        for i, existing in enumerate(canonical):
            if paper.doi and existing.doi and paper.doi == existing.doi:
                match_idx = i
            elif paper.source_ids.get("arxiv") and paper.source_ids.get("arxiv") == existing.source_ids.get("arxiv"):
                match_idx = i
            else:
                a, b = normalize_title(paper.title), normalize_title(existing.title)
                same_year = not paper.year or not existing.year or abs(paper.year - existing.year) <= 1
                if a and b and same_year and fuzz.ratio(a, b) >= 93:
                    match_idx = i
            if match_idx is not None:
                break
        if match_idx is None:
            canonical.append(paper)
        else:
            canonical[match_idx] = _merge(canonical[match_idx], paper)
    return canonical


def rank_papers(question: str, papers: list[Paper], reranker: Reranker) -> list[Paper]:
    """Score = semantic relevance (cross-encoder) + small priors for analysability."""
    if not papers:
        return []
    docs = [f"{p.title}. {(p.abstract or '')[:1200]}" for p in papers]
    rel = reranker.score(question, docs)
    max_cites = max((p.citation_count or 0) for p in papers) or 1
    ranked = []
    for p, r in zip(papers, rel):
        score = 0.72 * r
        score += 0.10 if p.abstract else 0.0
        score += 0.08 if (p.pdf_url or p.is_open_access) else 0.0
        score += 0.06 * math.log1p(p.citation_count or 0) / math.log1p(max_cites)
        score += 0.04 * min(1, len(p.sources) - 1)
        ranked.append(p.model_copy(update={"relevance": round(float(score), 4)}))
    ranked.sort(key=lambda p: p.relevance, reverse=True)
    return ranked


@dataclass
class DiscoveryResult:
    candidates: list[Paper]
    per_source: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def discover(queries: list[str], sources: list[LiteratureSource], limit_per_source: int) -> DiscoveryResult:
    """Query every source (in parallel across sources) and deduplicate the union."""
    result = DiscoveryResult(candidates=[])
    collected: list[Paper] = []

    def run(source: LiteratureSource) -> tuple[str, list[Paper], list[str]]:
        found: list[Paper] = []
        warns: list[str] = []
        for q in queries:
            try:
                found.extend(source.search(q, limit_per_source))
            except SourceError as exc:
                warns.append(f"{source.name}: {exc}")
                break  # don't hammer a source that is failing / rate limiting
            except Exception as exc:  # noqa: BLE001
                log.exception("unexpected error in %s", source.name)
                warns.append(f"{source.name}: unexpected error ({exc.__class__.__name__})")
                break
        return source.name, found, warns

    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as pool:
        for name, found, warns in pool.map(run, sources):
            result.per_source[name] = len(found)
            result.warnings.extend(warns)
            collected.extend(found)
    result.candidates = deduplicate(collected)
    return result
