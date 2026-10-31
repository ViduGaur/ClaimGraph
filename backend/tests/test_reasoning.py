"""Relationship classification, conflict analysis, graph construction, landscape."""
from app.graph.builder import build_graph, serialize_graph
from app.providers.models import EmbeddingReranker, HashingEmbedder, ModelBundle
from app.providers.llm import BudgetedLLM
from app.reasoning.conflicts import analyze_conflicts
from app.reasoning.landscape import LandscapeBuilder
from app.reasoning.relations import PairCandidate, RelationClassifier, compare_contexts, context_divergence
from app.schemas.domain import ClaimContext, ClaimType, ConflictVerdict, Polarity, RelationType
from tests.conftest import FixedNLI, make_claim, make_paper


def classifier(entail: float, contra: float) -> RelationClassifier:
    emb = HashingEmbedder()
    return RelationClassifier(ModelBundle(emb, EmbeddingReranker(emb), FixedNLI(entail, contra), []), BudgetedLLM(None, 0),
                              pair_threshold=0.72)  # calibrated like the default bge embeddings


def test_compare_contexts_statuses():
    a = ClaimContext(datasets=["CIFAR-10"], data_scale="small", models=["ViT", "ResNet"], conditions=["trained from scratch"])
    b = ClaimContext(datasets=["JFT-300M"], data_scale="web-scale", models=["DeiT", "CNN"], conditions=["large-scale pre-training", "fine-tuning"])
    f = {x.factor: x.status for x in compare_contexts(a, b)}
    assert f["dataset"] == "different"
    assert f["data scale"] == "different"
    assert f["model / method"] == "same"  # compared at family level (transformer vs cnn)
    assert f["experimental conditions"] == "different"
    assert f["evaluation metric"] == "unknown"  # unstated ≠ different


def test_divergence_ignores_unknown_and_discounts_inherited():
    same = compare_contexts(ClaimContext(datasets=["A"]), ClaimContext(datasets=["A"]))
    assert context_divergence(same) == 0.0
    assert context_divergence(compare_contexts(ClaimContext(), ClaimContext())) is None
    diff = compare_contexts(ClaimContext(datasets=["A"], data_scale="small"), ClaimContext(datasets=["B"], data_scale="small"),
                            a_inherited=["datasets"])
    full = compare_contexts(ClaimContext(datasets=["A"], data_scale="small"), ClaimContext(datasets=["B"], data_scale="small"))
    assert context_divergence(diff) < context_divergence(full)


def _pair(ctx_a: dict, ctx_b: dict, pol_b=Polarity.NEGATIVE, sim=0.85):
    a = make_claim("a", "p1", "ViTs outperform CNNs on image classification benchmarks.", baselines=["CNN"], **ctx_a)
    b = make_claim("b", "p2", "ViTs underperform CNNs on image classification benchmarks.", polarity=pol_b, baselines=["CNN"], **ctx_b)
    return a, b, PairCandidate(a, b, sim)


def test_opposing_claims_under_different_conditions_qualify():
    a, b, pair = _pair({"datasets": ["JFT-300M"], "data_scale": "web-scale", "conditions": ["large-scale pre-training"]},
                       {"datasets": ["CIFAR-10"], "data_scale": "small", "conditions": ["trained from scratch"]})
    rels = classifier(0.02, 0.9).classify([pair])
    assert rels[0].relation == RelationType.QUALIFIES
    conflicts = analyze_conflicts(rels, {"a": a, "b": b})
    assert conflicts[0].verdict == ConflictVerdict.CONTEXT_DEPENDENT
    assert "may not actually be contradictory" in conflicts[0].explanation
    assert {f.factor for f in conflicts[0].differing_factors} >= {"dataset", "data scale"}


def test_opposing_claims_under_same_conditions_contradict():
    same = {"datasets": ["ImageNet-1k"], "data_scale": "medium", "metrics": ["top-1 accuracy"], "task": ["image classification"]}
    a, b, pair = _pair(same, same)
    rels = classifier(0.02, 0.9).classify([pair])
    assert rels[0].relation == RelationType.CONTRADICTS and not rels[0].uncertain
    conflicts = analyze_conflicts(rels, {"a": a, "b": b})
    assert conflicts[0].verdict == ConflictVerdict.GENUINE
    assert "substantially similar conditions" in conflicts[0].explanation


def test_opposing_claims_without_context_are_not_forced():
    a, b, pair = _pair({}, {})
    rels = classifier(0.02, 0.9).classify([pair])
    assert rels[0].uncertain
    conflicts = analyze_conflicts(rels, {"a": a, "b": b})
    assert conflicts[0].verdict == ConflictVerdict.INSUFFICIENT_EVIDENCE
    assert conflicts[0].confidence <= 0.45


def test_agreeing_claims_support():
    same = {"datasets": ["ImageNet-1k"], "data_scale": "medium"}
    _, _, pair = _pair(same, same, pol_b=Polarity.POSITIVE)
    rels = classifier(0.1, 0.05).classify([pair])
    assert rels[0].relation == RelationType.SUPPORTS


def test_agreement_in_new_setting_extends():
    _, _, pair = _pair({"datasets": ["ImageNet-1k"], "data_scale": "medium"}, {"datasets": ["CIFAR-10"], "data_scale": "small"},
                       pol_b=Polarity.POSITIVE)
    rels = classifier(0.8, 0.02).classify([pair])
    assert rels[0].relation == RelationType.EXTENDS


def test_nli_contradiction_does_not_override_agreeing_stance():
    _, _, pair = _pair({}, {}, pol_b=Polarity.POSITIVE)
    rels = classifier(0.0, 0.75).classify([pair])
    assert not rels or rels[0].relation != RelationType.CONTRADICTS


def test_unrelated_claims_yield_no_relation():
    a = make_claim("a", "p1", "Soil bacteria increase wheat yield.", subject=None, polarity=Polarity.NONE)
    b = make_claim("b", "p2", "Transformers scale with compute.", subject=None, polarity=Polarity.NONE)
    assert classifier(0.1, 0.1).classify([PairCandidate(a, b, 0.5)]) == []


def test_graph_construction_and_stats():
    papers = [make_paper("p1"), make_paper("p2")]
    a, b, pair = _pair({"data_scale": "small", "datasets": ["X"]}, {"data_scale": "web-scale", "datasets": ["Y"]})
    rels = classifier(0.02, 0.9).classify([pair])
    extra = make_claim("a2", "p1", "A second claim from the same paper about ViTs.")
    g = build_graph(papers, [a, b, extra], rels)
    assert g.number_of_nodes() == 5
    model = serialize_graph(g)
    kinds = sorted(e.kind for e in model.edges)
    assert len({e.id for e in model.edges}) == len(model.edges)  # ids must be unique (Cytoscape rejects duplicates)
    assert kinds == ["MAKES", "MAKES", "MAKES", "QUALIFIES"]
    assert model.stats["relationships"] == 1
    assert model.stats["isolated_claims"] == 1
    assert set(model.stats["bridging_papers"]) == {"p1", "p2"}


def test_landscape_gaps_are_hedged(models):
    same = {"datasets": ["ImageNet-1k"], "data_scale": "medium"}
    a, b, pair = _pair(same, same)
    rels = classifier(0.02, 0.9).classify([pair])
    conflicts = analyze_conflicts(rels, {"a": a, "b": b})
    lim1 = make_claim("l1", "p1", "However, training remains unstable on small datasets.", ctype=ClaimType.LIMITATION)
    lim2 = make_claim("l2", "p2", "However, training remains unstable on small datasets.", ctype=ClaimType.LIMITATION)
    land = LandscapeBuilder(models, BudgetedLLM(None, 0)).build("q", [make_paper("p1"), make_paper("p2")], [a, b, lim1, lim2], rels, conflicts)
    assert land.gaps
    assert all(g.title.startswith("Potential research gap") for g in land.gaps)
    assert all(g.evidence_claim_ids for g in land.gaps)
    signals = {g.signal for g in land.gaps}
    assert "repeated_limitation" in signals and "unresolved_conflict" in signals
    assert land.disagreements and land.unresolved_questions


def test_landscape_quotes_verbatim_text_and_counts_grammatically(models):
    same = {"datasets": ["ImageNet-1k"], "data_scale": "medium"}
    a, b, pair = _pair(same, same)
    a = a.model_copy(update={"normalized_text": "rewritten text that must never be quoted"})
    rels = classifier(0.02, 0.9).classify([pair])
    conflicts = analyze_conflicts(rels, {"a": a, "b": b})
    land = LandscapeBuilder(models, BudgetedLLM(None, 0)).build("q", [make_paper("p1"), make_paper("p2")], [a, b], rels, conflicts)
    shown = " ".join([*land.unresolved_questions, *(g.title for g in land.gaps)])
    assert "rewritten text" not in shown and a.claim_text[:40] in shown
    assert "Of 1 apparent disagreement," in land.summary
    assert " 1 occurs " in land.summary or " 1 lacks " in land.summary or " 1 is " in land.summary


def test_uncertain_conflict_names_weak_link_not_missing_context():
    a, b, pair = _pair({"datasets": ["JFT-300M"], "data_scale": "web-scale"}, {"datasets": ["CIFAR-10"], "data_scale": "small"})
    rel = classifier(0.02, 0.9).classify([pair])[0].model_copy(update={"relation": RelationType.CONTRADICTS, "uncertain": True})
    (cf,) = analyze_conflicts([rel], {"a": a, "b": b})
    assert cf.verdict == ConflictVerdict.INSUFFICIENT_EVIDENCE
    assert "low-confidence" in cf.headline and "data scale" in cf.explanation
