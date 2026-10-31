"""End-to-end: the whole pipeline on the bundled demo corpus with offline models.

Checks the invariants that make ClaimGraph trustworthy rather than exact numbers:
every claim is grounded in verbatim source text, every relationship and conflict
refers to real claims from different papers, and gaps are hedged.
"""
import pytest

from app.pipelines.analysis import AnalysisPipeline
from app.schemas.api import AnalyzeRequest
from app.schemas.domain import ConflictVerdict, RelationType


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    from app.core.config import Settings
    from app.providers.models import EmbeddingReranker, HashingEmbedder, LexicalNLI, ModelBundle

    tmp = tmp_path_factory.mktemp("e2e")
    s = Settings(demo_mode=True, data_dir=tmp, embedding_backend="hashing", nli_backend="lexical", llm_provider="none", _env_file=None)
    emb = HashingEmbedder()
    stages = []
    res = AnalysisPipeline(s, progress=lambda k, st, d: stages.append((k, st)),
                           models=ModelBundle(emb, EmbeddingReranker(emb), LexicalNLI(), [])).run(
        AnalyzeRequest(question="Do vision transformers need large-scale pre-training to outperform convolutional networks?"))
    res._stages = stages  # type: ignore[attr-defined]
    return res


def test_all_stages_reported(result):
    done = [k for k, st in result._stages if st == "done"]
    assert done == ["search", "select", "acquire", "extract", "compare", "graph", "landscape"]


def test_claims_are_grounded(result):
    papers = {p.paper_id: p for p in result.papers}
    assert len(result.claims) >= 10
    for c in result.claims:
        assert c.paper_id in papers
        assert c.claim_text in c.evidence[0].text
        assert c.claim_text in (papers[c.paper_id].abstract or "")  # demo corpus = abstracts
        assert 0 < c.confidence <= 0.95


def test_relationships_are_cross_paper_and_explained(result):
    claims = {c.claim_id: c for c in result.claims}
    assert result.relationships
    for r in result.relationships:
        a, b = claims[r.target_claim_id], claims[r.source_claim_id]
        assert a.paper_id != b.paper_id
        assert r.reasoning and r.factors
        assert r.relation in RelationType


def test_conflicts_link_to_relationships(result):
    rel_ids = {r.relation_id for r in result.relationships}
    for c in result.conflicts:
        assert c.relation_id in rel_ids
        assert c.verdict in ConflictVerdict
        if c.verdict == ConflictVerdict.CONTEXT_DEPENDENT:
            assert c.differing_factors


def test_graph_matches_results(result):
    node_ids = {n.id for n in result.graph.nodes}
    assert {p.paper_id for p in result.papers} <= node_ids
    assert {c.claim_id for c in result.claims} <= node_ids
    rel_edges = [e for e in result.graph.edges if e.kind != "MAKES"]
    assert len(rel_edges) == len(result.relationships)


def test_landscape_is_hedged(result):
    land = result.landscape
    assert land.summary.startswith("ClaimGraph analysed")
    for g in land.gaps:
        assert g.title.startswith("Potential research gap")
        assert g.evidence_claim_ids
