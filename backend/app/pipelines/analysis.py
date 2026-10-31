"""The ClaimGraph pipeline.

question → query understanding → literature discovery → paper ranking →
open-access acquisition → text extraction → relevant passage retrieval →
claim extraction → normalisation → claim comparison → relationship
classification → contextual disagreement analysis → claim graph → landscape

DEMO_MODE swaps only the literature source (a curated snapshot of real arXiv
papers) and skips network downloads; every other stage is identical.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import Settings
from app.extraction.acquisition import Acquirer, PaperText
from app.extraction.claims import ClaimExtractor
from app.extraction.text import Section
from app.graph.builder import build_graph, serialize_graph
from app.literature.discovery import discover, rank_papers
from app.literature.http import PoliteClient
from app.literature.sources import (
    ArxivSource,
    EuropePMCSource,
    FixtureSource,
    LiteratureSource,
    OpenAlexSource,
    SemanticScholarSource,
)
from app.providers.llm import BudgetedLLM, build_llm
from app.providers.models import ModelBundle, get_models
from app.reasoning.alignment import LLMPairAnalyzer
from app.reasoning.conflicts import analyze_conflicts
from app.reasoning.explanations import explain_conflicts
from app.reasoning.landscape import LandscapeBuilder
from app.reasoning.query import is_biomedical, understand_query
from app.reasoning.relations import RelationClassifier, auto_threshold, candidate_pairs
from app.schemas.api import AnalysisResult, AnalyzeRequest, RunInfo
from app.schemas.domain import Claim, Paper

log = logging.getLogger(__name__)
DEMO_CORPUS = Path(__file__).resolve().parents[1] / "demo" / "corpus.json"

# callback(stage_key, status, detail)
ProgressFn = Callable[[str, str, str | None], None]
WarnFn = Callable[[str, str], None]


class PipelineError(RuntimeError):
    """Unrecoverable failure (e.g. no papers found); message is user-facing."""


class AnalysisPipeline:
    def __init__(self, settings: Settings, progress: ProgressFn | None = None, warn: WarnFn | None = None,
                 models: ModelBundle | None = None, sources: list[LiteratureSource] | None = None,
                 client: PoliteClient | None = None):
        self.settings = settings
        self.progress = progress or (lambda *_: None)
        self.warn = warn or (lambda stage, msg: log.warning("[%s] %s", stage, msg))
        self._models = models
        self._sources = sources
        self._client = client

    # ------------------------------------------------------------------ wiring
    def _build_sources(self, demo: bool, biomedical: bool, client: PoliteClient) -> list[LiteratureSource]:
        if self._sources is not None:
            return self._sources
        if demo:
            return [FixtureSource(DEMO_CORPUS)]
        s = self.settings
        names = list(s.source_list)
        if biomedical and "europepmc" not in names:
            names.append("europepmc")
        registry: dict[str, Callable[[], LiteratureSource]] = {
            "openalex": lambda: OpenAlexSource(client, s.openalex_api_key, s.contact_email),
            "semantic_scholar": lambda: SemanticScholarSource(client, s.semantic_scholar_api_key),
            "arxiv": lambda: ArxivSource(client),
            "europepmc": lambda: EuropePMCSource(client),
        }
        return [registry[n]() for n in names if n in registry]

    # ------------------------------------------------------------------ run
    def run(self, req: AnalyzeRequest) -> AnalysisResult:
        s = self.settings
        t0 = time.monotonic()
        demo = bool(req.demo) if req.demo is not None else s.demo_mode
        max_papers = req.max_papers or s.max_papers
        client = self._client or PoliteClient(s.contact_email, s.http_timeout_s)

        llm, llm_warning = build_llm(s, req.use_llm)
        if llm_warning:
            self.warn("setup", llm_warning)
        models = self._models or get_models(s)
        for note in models.notes:
            self.warn("setup", note)

        try:
            # 1. search ---------------------------------------------------------
            self.progress("search", "running", "Understanding the question")
            plan = understand_query(req.question, llm)
            sources = self._build_sources(demo, is_biomedical(plan), client)
            self.progress("search", "running", f"Querying {', '.join(src.name for src in sources)}")
            found = discover(plan.search_queries, sources, s.max_candidates_per_source)
            for w in found.warnings:
                self.warn("search", w)
            if not found.candidates:
                raise PipelineError("No papers were found. " + ("; ".join(found.warnings) or "Try rephrasing the question."))
            per_source = ", ".join(f"{k}: {v}" for k, v in found.per_source.items())
            self.progress("search", "done", f"{len(found.candidates)} unique papers ({per_source})")

            # 2. select -----------------------------------------------------------
            self.progress("select", "running", "Ranking candidates with a cross-encoder")
            ranked = rank_papers(req.question, found.candidates, models.reranker)
            analysable = [p for p in ranked if p.abstract or p.pdf_url or p.source_ids.get("arxiv")]
            selected = analysable[:max_papers]
            if len(selected) < 2:
                raise PipelineError("Fewer than two analysable papers were found; cross-paper analysis needs at least two.")
            self.progress("select", "done", f"Selected {len(selected)} of {len(ranked)} papers")

            # 3. acquire ----------------------------------------------------------
            docs = self._acquire(selected, demo, client)

            # 4. extract ----------------------------------------------------------
            claims = self._extract(req.question, docs, models, llm)
            papers = [d.paper for d in docs]
            paper_map = {p.paper_id: p for p in papers}

            # 5. compare ----------------------------------------------------------
            self.progress("compare", "running", "Pairing claims and running NLI")
            threshold = auto_threshold(models, s.pair_similarity_threshold)
            pairs = candidate_pairs(claims, paper_map, models, threshold, s.max_pairs)
            analyzer = LLMPairAnalyzer(paper_map, s.llm_concurrency) if llm.enabled and s.llm_pair_analysis else None
            explain = llm.enabled and s.llm_conflict_explanations
            # keep LLM calls back for the conflict explanations and the summary
            reserve = 1 + (s.llm_explanation_budget if explain else 0)
            classifier = RelationClassifier(models, llm, pair_threshold=threshold, analyzer=analyzer, llm_reserve=reserve)
            relationships = classifier.classify(pairs)
            claim_map = {c.claim_id: c for c in claims}
            conflicts = analyze_conflicts(relationships, claim_map)
            # the verdicts are final at this point; the LLM only words them, from the decision's own evidence
            conflicts = explain_conflicts(conflicts, relationships, claim_map, llm, concurrency=s.llm_concurrency,
                                          limit=max(0, llm.remaining - 1) if explain else 0)
            self.progress("compare", "done", f"{len(pairs)} candidate pairs → {len(relationships)} relationships, "
                                              f"{len(conflicts)} apparent conflicts")

            # 6. graph ------------------------------------------------------------
            self.progress("graph", "running", None)
            graph = serialize_graph(build_graph(papers, claims, relationships))
            self.progress("graph", "done", f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")

            # 7. landscape --------------------------------------------------------
            self.progress("landscape", "running", None)
            landscape = LandscapeBuilder(models, llm).build(req.question, papers, claims, relationships, conflicts)
            self.progress("landscape", "done", f"{len(landscape.consensus)} consensus, {len(landscape.disagreements)} disputed themes, "
                                                f"{len(landscape.gaps)} potential gaps")
            for f in llm.failures:
                self.warn("llm", f)

            return AnalysisResult(
                query_plan=plan,
                papers=papers,
                claims=claims,
                relationships=relationships,
                conflicts=conflicts,
                graph=graph,
                landscape=landscape,
                run=RunInfo(
                    demo_mode=demo,
                    llm_provider=s.llm_provider_name if llm.enabled else "none",
                    llm_model=s.llm_model_name if llm.enabled else None,
                    llm_calls=llm.calls,
                    embedding_backend=f"{models.embedder.name}" + (f" ({s.embedding_model})" if models.embedder.name == "fastembed" else ""),
                    nli_backend=f"{models.nli.name}" + (f" ({s.nli_model})" if models.nli.name == "onnx" else ""),
                    sources=[src.name for src in sources],
                    duration_s=round(time.monotonic() - t0, 2),
                ),
            )
        finally:
            if self._client is None:
                client.close()

    # ------------------------------------------------------------------ stages
    def _acquire(self, selected: list[Paper], demo: bool, client: PoliteClient) -> list[PaperText]:
        s = self.settings
        self.progress("acquire", "running", f"Locating open-access text for {len(selected)} papers")
        if demo:
            docs = [PaperText(p.model_copy(update={"text_source": "abstract", "acquisition_note": "demo corpus: arXiv abstract (CC0)"}),
                              [Section("abstract", p.abstract or "")]) for p in selected]
        else:
            acquirer = Acquirer(client, s.papers_dir, s.contact_email, s.max_pdf_mb, s.download_pdfs)

            def safe(p: Paper) -> PaperText:
                try:
                    return acquirer.acquire(p)
                except Exception as exc:  # noqa: BLE001 — isolate per-paper failures
                    log.exception("acquisition failed for %s", p.paper_id)
                    fallback = [Section("abstract", p.abstract)] if p.abstract else []
                    return PaperText(p.model_copy(update={"text_source": "abstract" if p.abstract else "none",
                                                          "acquisition_note": f"acquisition error: {exc.__class__.__name__}"}), fallback)

            with ThreadPoolExecutor(max_workers=4) as pool:
                docs = list(pool.map(safe, selected))
        for d in docs:
            if d.paper.text_source != "full_text" and not demo:
                self.warn("acquire", f"{d.paper.citation}: {d.paper.acquisition_note}")
        docs = [d for d in docs if d.sections]
        full = sum(1 for d in docs if d.paper.text_source == "full_text")
        self.progress("acquire", "done", f"{full} full text, {len(docs) - full} abstract only")
        if len(docs) < 2:
            raise PipelineError("Could not obtain text for at least two papers.")
        return docs

    def _extract(self, question: str, docs: list[PaperText], models: ModelBundle, llm: BudgetedLLM) -> list[Claim]:
        s = self.settings
        self.progress("extract", "running", "Retrieving relevant passages and extracting claims")
        extractor = ClaimExtractor(models, llm, s.max_claims_per_paper, s.llm_for_extraction,
                                   llm_selection=s.llm_claim_selection)
        qvec = models.embedder.embed([question])[0]
        claims: list[Claim] = []
        for i, doc in enumerate(docs, 1):
            try:
                found, warnings = extractor.extract(question, qvec, doc)
            except Exception as exc:  # noqa: BLE001 — one paper must not fail the analysis
                log.exception("claim extraction failed for %s", doc.paper.paper_id)
                self.warn("extract", f"{doc.paper.citation}: extraction failed ({exc.__class__.__name__})")
                continue
            for w in warnings:
                self.warn("extract", w)
            claims.extend(found)
            self.progress("extract", "running", f"{i}/{len(docs)} papers · {len(claims)} claims")
        if not claims:
            raise PipelineError("No claims could be extracted from the selected papers.")
        self.progress("extract", "done", f"{len(claims)} claims from {len(docs)} papers")
        return claims
