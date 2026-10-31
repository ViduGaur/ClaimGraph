"""Research-version components: ablation switches, the grounded LLM pair analyser, LLM claim
selection, the response cache, and the evaluation harness (labels, metrics, systems)."""
from __future__ import annotations

import json

import pytest

from app.evaluation import annotation as ann
from app.evaluation import metrics as M
from app.evaluation.corpus import Topic
from app.evaluation.experiment import NO_GATE, Systems, TopicData, final_label, pair_id, predict, predict_full, two_level
from app.extraction.acquisition import PaperText
from app.extraction.claims import ClaimExtractor
from app.extraction.text import Section
from app.providers.llm import BudgetedLLM, CachedProvider, LLMError, LLMProvider
from app.providers.models import EmbeddingReranker, HashingEmbedder, ModelBundle
from app.reasoning.alignment import GroundingStats, LLMPairAnalyzer, view_from_output
from app.reasoning.relations import PairCandidate, RelationClassifier, RuleConfig
from app.schemas.domain import Polarity, RelationType
from tests.conftest import FixedNLI, make_claim, make_paper

NLI_CON = {"entailment": 0.02, "contradiction": 0.9, "neutral": 0.08}
NLI_NEU = {"entailment": 0.02, "contradiction": 0.02, "neutral": 0.96}


class ScriptedLLM(LLMProvider):
    """Returns queued responses; an Exception in the queue is raised."""

    name = "scripted"

    def __init__(self, responses):
        super().__init__("test-model")
        self.responses = list(responses)
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def complete_json(self, system, user, schema, max_tokens=400):
        self.prompts.append(user)
        out = self.responses.pop(0) if self.responses else {}
        if isinstance(out, Exception):
            raise out
        return out


def clf(entail: float, contra: float, **kw) -> RelationClassifier:
    emb = HashingEmbedder()
    return RelationClassifier(ModelBundle(emb, EmbeddingReranker(emb), FixedNLI(entail, contra), []), kw.pop("llm", BudgetedLLM(None, 0)),
                              pair_threshold=0.12, **kw)


def opposing_pair(**ctx_b):
    a = make_claim("a", "p1", "ViT outperforms CNN on ImageNet.", datasets=["ImageNet-1k"], data_scale="large", baselines=["CNN"])
    b = make_claim("b", "p2", "ViT underperforms CNN on small datasets.", polarity=Polarity.NEGATIVE, baselines=["CNN"],
                   **(ctx_b or {"datasets": ["CIFAR-10"], "data_scale": "small"}))
    return PairCandidate(a, b, 0.6)


# --------------------------------------------------------------------------- ablation switches
def test_without_context_gate_every_conflict_is_a_contradiction():
    pair = opposing_pair()
    full = clf(0.02, 0.9).decide(pair, NLI_CON, NLI_CON)
    no_gate = clf(0.02, 0.9, config=RuleConfig(use_context_gate=False)).decide(pair, NLI_CON, NLI_CON)
    assert full.relation == RelationType.QUALIFIES  # different data scale explains the conflict
    assert no_gate.relation == RelationType.CONTRADICTS and not no_gate.uncertain
    assert final_label(no_gate, NO_GATE, {"a": pair.a, "b": pair.b}) == "CONTRADICTS"


def test_without_stance_a_weak_nli_conflict_is_missed():
    pair = opposing_pair()
    weak = {"entailment": 0.02, "contradiction": 0.3, "neutral": 0.68}
    assert clf(0.02, 0.3).decide(pair, weak, weak) is not None  # opposing stance + some NLI contradiction
    assert clf(0.02, 0.3, config=RuleConfig(use_stance=False)).decide(pair, weak, weak) is None


# --------------------------------------------------------------------------- grounded LLM pair analysis
def llm_output(stance="oppose", **factor):
    base = {"factor": "experimental conditions", "a_value": "pre-trained on JFT", "a_quote": "pre-trained on JFT-300M",
            "b_value": "from scratch", "b_quote": "trained from scratch on ImageNet", "status": "different", "material": True}
    return {"subject_a": "ViT versus CNNs", "outcome_a": "accuracy", "subject_b": "ViT versus CNNs", "outcome_b": "accuracy",
            "same_relationship": stance != "unrelated", "conclusion_a": "ViT is better", "conclusion_b": "ViT is worse",
            "stance": stance, "factors": [{**base, **factor}]}


@pytest.fixture
def llm_pair():
    a = make_claim("a", "p1", "ViT beats CNNs when pre-trained on JFT-300M.", subject=None, polarity=Polarity.NONE)
    b = make_claim("b", "p2", "ViT is worse than CNNs when trained from scratch on ImageNet.", subject=None, polarity=Polarity.NONE)
    papers = {"p1": make_paper("p1").model_copy(update={"abstract": "We study large-scale pre-training on 300M images."}),
              "p2": make_paper("p2")}
    return PairCandidate(a, b, 0.5), papers


def factor(view, name="experimental conditions"):
    return next(f for f in view.factors if f.factor == name)


def test_llm_view_drives_the_same_rules(llm_pair):
    pair, papers = llm_pair
    view = view_from_output(llm_output(), pair.a, pair.b, papers)
    assert view.stance == "oppose" and view.confident and factor(view).status == "different"
    rel = clf(0.0, 0.0).decide(pair, NLI_NEU, NLI_NEU, view)  # NLI sees nothing; the semantic stance suffices
    assert rel.relation == RelationType.QUALIFIES and rel.decided_by == "rules+llm" and rel.llm_note
    assert final_label(rel, RuleConfig(), {"a": pair.a, "b": pair.b}) == "QUALIFIES"


def test_ungrounded_quote_is_dropped_so_unknown_is_never_different(llm_pair):
    pair, papers = llm_pair
    stats = GroundingStats()
    view = view_from_output(llm_output(b_quote="evaluated only on CIFAR-10 with heavy augmentation"), pair.a, pair.b, papers, stats)
    assert factor(view).status == "unknown" and factor(view).b == []
    assert stats.dropped == 1 and stats.grounded_local == 1
    rel = clf(0.0, 0.0).decide(pair, NLI_NEU, NLI_NEU, view)
    assert rel.relation == RelationType.CONTRADICTS and rel.uncertain  # conflict, but no comparable context → abstain
    assert final_label(rel, RuleConfig(), {"a": pair.a, "b": pair.b}) == M.ABSTAIN  # uncertain CONTRADICTS, not asserted
    ungated = view_from_output(llm_output(b_quote="evaluated only on CIFAR-10 with heavy augmentation"), pair.a, pair.b, papers,
                               enforce_grounding=False)
    assert factor(ungated).status == "different"  # what the 'no grounding' ablation would have believed


def test_one_sided_and_immaterial_factors(llm_pair):
    pair, papers = llm_pair
    one_sided = view_from_output(llm_output(b_value=None, b_quote=None, status="different"), pair.a, pair.b, papers)
    assert factor(one_sided).status == "unknown"
    stats = GroundingStats()
    immaterial = view_from_output(llm_output(material=False), pair.a, pair.b, papers, stats)
    assert factor(immaterial).status == "overlap" and stats.downgraded_immaterial == 1
    rel = clf(0.0, 0.0).decide(pair, NLI_NEU, NLI_NEU, immaterial)
    assert rel.relation == RelationType.CONTRADICTS and not rel.uncertain  # an incidental difference does not excuse a conflict


def test_value_grounded_only_in_abstract_is_inherited(llm_pair):
    pair, papers = llm_pair
    pair.a.section = "results"  # the claim is not from the abstract, so abstract-level context is paper-level
    view = view_from_output(llm_output(a_value="300M images", a_quote="large-scale pre-training on 300M images"), pair.a, pair.b, papers)
    assert factor(view).status == "different" and factor(view).inherited


def test_unrelated_stance_ends_analysis_and_bad_output_is_rejected(llm_pair):
    pair, papers = llm_pair
    view = view_from_output(llm_output(stance="unrelated"), pair.a, pair.b, papers)
    assert view.comparable is False
    assert clf(0.0, 0.9).decide(pair, NLI_CON, NLI_CON, view) is None  # even a confident NLI contradiction
    assert view_from_output({"stance": "maybe", "factors": []}, pair.a, pair.b, papers) is None


def test_llm_failure_falls_back_to_rule_view(llm_pair):
    pair, papers = llm_pair
    llm = BudgetedLLM(ScriptedLLM([LLMError("boom"), llm_output()]), max_calls=5)
    analyzer = LLMPairAnalyzer(papers, concurrency=1)
    classifier = clf(0.0, 0.0, llm=llm, analyzer=analyzer)
    views = classifier.views([pair, pair], [(NLI_NEU, NLI_NEU)] * 2)
    assert [v.source for v in views] == ["rules", "llm"] and analyzer.stats.failures == 1
    assert len(llm.failures) == 1


def test_llm_budget_reserve_limits_analysed_pairs(llm_pair):
    pair, papers = llm_pair
    provider = ScriptedLLM([llm_output()] * 5)
    classifier = clf(0.0, 0.0, llm=BudgetedLLM(provider, max_calls=3), analyzer=LLMPairAnalyzer(papers, concurrency=1), llm_reserve=1)
    views = classifier.views([pair] * 4, [(NLI_NEU, NLI_NEU)] * 4)
    assert sum(v.source == "llm" for v in views) == 2 and len(provider.prompts) == 2


# --------------------------------------------------------------------------- cache
def test_cached_provider_replays_and_never_caches_failures(tmp_path):
    inner = ScriptedLLM([LLMError("down"), {"ok": 1}, {"ok": 2}])
    cached = CachedProvider(inner, tmp_path / "cache.jsonl")
    with pytest.raises(LLMError):
        cached.complete_json("s", "u", {})
    assert cached.complete_json("s", "u", {}) == {"ok": 1}
    assert cached.complete_json("s", "u", {}) == {"ok": 1} and (cached.hits, cached.misses) == (1, 1)
    assert cached.complete_json("s", "other", {}) == {"ok": 2}
    reloaded = CachedProvider(ScriptedLLM([]), tmp_path / "cache.jsonl")  # a fresh process replays from disk
    assert reloaded.complete_json("s", "u", {}) == {"ok": 1} and reloaded.hits == 1


# --------------------------------------------------------------------------- LLM claim selection
ABSTRACT = ("Recently, transformers have been widely adopted for vision. We show that ViT outperforms ResNet on ImageNet by 2%. "
            "We propose a new training recipe for transformers. Our results indicate that augmentation improves accuracy significantly. "
            "However, ViT fails to converge on small datasets without regularization.")


def extract_with(provider, models, **kw):
    doc = PaperText(make_paper("p1").model_copy(update={"abstract": ABSTRACT, "text_source": "abstract"}), [Section("abstract", ABSTRACT)])
    extractor = ClaimExtractor(models, BudgetedLLM(provider, 5), 6, llm_selection=True, **kw)
    return extractor.extract("Do vision transformers beat CNNs?", models.embedder.embed(["Do vision transformers beat CNNs?"])[0], doc)[0], doc


def test_llm_selection_only_keeps_verbatim_candidates(models):
    provider = ScriptedLLM([{"keep": [{"n": 0, "type": "comparison"}, {"n": 99, "type": "finding"}, {"n": 0, "type": "finding"}]}])
    claims, doc = extract_with(provider, models)
    assert len(claims) == 1 and claims[0].extraction_method == "llm_selected"
    assert claims[0].claim_text in doc.sections[0].text  # the LLM can choose, never rewrite
    assert "[0]" in provider.prompts[0]


def test_llm_selection_failure_falls_back_to_cue_ranking(models):
    claims, _ = extract_with(ScriptedLLM([LLMError("timeout")]), models)
    assert claims and all(c.extraction_method == "heuristic" for c in claims)


# --------------------------------------------------------------------------- annotation
def test_derive_label_covers_the_decision_tree():
    assert ann.derive_label("unrelated", None) == "UNRELATED"
    assert ann.derive_label("agree", "comparable") == "SUPPORTS" and ann.derive_label("agree", "not_stated") == "SUPPORTS"
    assert ann.derive_label("agree", "different") == "EXTENDS"
    assert ann.derive_label("partial", "comparable") == "QUALIFIES"
    # two levels: an opposing pair without stated context is CONTRADICTS at the relation level, verdict insufficient
    assert [ann.derive_label("oppose", c) for c in ann.CONTEXTS] == ["CONTRADICTS", "QUALIFIES", "CONTRADICTS"]
    assert [ann.derive_verdict("oppose", c) for c in ann.CONTEXTS] == ["genuine", "context_dependent", "insufficient"]
    assert ann.derive_verdict("partial", "different") is None and ann.derive_verdict("agree", "comparable") is None
    assert ann.derive_label("oppose", None) is None


def answer(stance, context=None):
    return {"stance": stance, "context": context, "bad_claim": False, "note": ""}


def test_gold_needs_agreement_or_adjudication():
    labels = {"i1": {"x": answer("oppose", "different"), "y": answer("oppose", "different")},
              "i2": {"x": answer("oppose", "comparable"), "y": answer("agree", "comparable")},
              "i3": {"x": answer("unrelated")}}
    gold, unresolved = ann.gold_labels(labels)
    assert gold["i1"]["label"] == "QUALIFIES" and gold["i3"]["source"] == "single" and unresolved == ["i2"]
    gold, unresolved = ann.gold_labels(labels, {"i2": answer("oppose", "comparable")})
    assert gold["i2"]["label"] == "CONTRADICTS" and gold["i2"]["source"] == "adjudicated" and not unresolved


def test_agreement_report_kappa():
    assert ann.cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert ann.cohen_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0
    labels = {f"i{k}": {"x": answer("agree", "comparable"), "y": answer("agree", "comparable" if k else "different")} for k in range(4)}
    report = ann.agreement_report(labels)
    assert report["pooled"]["stance"]["agreement"] == 1.0 and report["pooled"]["label"]["agreement"] == 0.75


# --------------------------------------------------------------------------- metrics
def ex(gold, pred, gold_verdict=None, pred_verdict=None, i=[0], **kw):  # noqa: B006 — counter for unique ids
    i[0] += 1
    return M.Example(f"i{i[0]}", "t", gold, pred, gold_verdict, pred_verdict, **kw)


def test_relation_metrics_treat_an_abstention_as_neither_right_nor_asserted():
    items = [ex("CONTRADICTS", "CONTRADICTS", "genuine", "genuine"), ex("QUALIFIES", "CONTRADICTS", "context_dependent", "genuine"),
             ex("UNRELATED", "CONTRADICTS", None, "genuine"), ex("CONTRADICTS", M.ABSTAIN, "genuine", "insufficient"),
             ex("QUALIFIES", "UNRELATED", "context_dependent", None), ex("SUPPORTS", None)]
    assert M.METRIC_FNS["fcr"](items) == pytest.approx(2 / 3)
    assert M.METRIC_FNS["false_con_per_100"](items) == pytest.approx(2 / 6)  # 2 false assertions among 6 pairs
    assert M.METRIC_FNS["con_precision"](items) == pytest.approx(1 / 3)
    assert M.METRIC_FNS["con_recall"](items) == pytest.approx(1 / 2)  # the abstained contradiction is a miss
    assert M.METRIC_FNS["accuracy"](items) == pytest.approx(1 / 6)
    assert M.METRIC_FNS["abstention"](items) == pytest.approx(1 / 6)
    assert M.METRIC_FNS["verdict_acc"](items) == pytest.approx(1 / 4)  # a missed or abstained conflict counts as wrong
    assert M.METRIC_FNS["verdict_acc_detected"](items) == pytest.approx(1 / 3)
    assert M.counts(items)["n_abstain"] == 1 and M.counts(items)["coverage"] == pytest.approx(5 / 6)
    assert M.METRIC_FNS["fcr"]([ex("QUALIFIES", "QUALIFIES")]) is None  # nothing asserted → undefined, not 0
    as_con = M.remap_abstentions(items, "CONTRADICTS")
    assert M.METRIC_FNS["con_recall"](as_con) == 1.0 and M.counts(as_con)["n_pred_con"] == 4
    assert M.confusion(items)["CONTRADICTS"][M.ABSTAIN] == 1


def test_design_weights_change_pool_estimates_but_not_sample_scores():
    # a conflict stratum sampled completely and a background stratum sampled 1 in 10
    items = [ex("QUALIFIES", "CONTRADICTS", weight=1.0, stratum="conflict") for _ in range(4)] + \
            [ex("UNRELATED", "UNRELATED", weight=10.0, stratum="background") for _ in range(4)]
    assert M.METRIC_FNS["false_con_per_100"](M.unweighted(items)) == pytest.approx(0.5)
    assert M.METRIC_FNS["false_con_per_100"](items) == pytest.approx(4 / 44)
    assert M.METRIC_FNS["accuracy"](items) == pytest.approx(40 / 44)


def test_bootstrap_interval_and_paired_difference():
    good = [ex("CONTRADICTS", "CONTRADICTS") for _ in range(30)] + [ex("QUALIFIES", "QUALIFIES") for _ in range(30)]
    bad = [M.Example(e.item_id, e.topic, e.gold, "CONTRADICTS") for e in good]
    lo, hi = M.bootstrap(bad, M.METRIC_FNS["fcr"], n_boot=300)
    assert 0.3 < lo < 0.5 < hi < 0.7
    d = M.paired_bootstrap(bad, good, M.METRIC_FNS["fcr"], n_boot=300)
    assert d["delta"] == pytest.approx(0.5) and d["lo"] > 0 and d["p"] < 0.05


# --------------------------------------------------------------------------- systems over pair records
def test_systems_are_pure_functions_of_a_pair_record(llm_pair):
    pair, papers = llm_pair
    rec = {"pair_id": pair_id("a", "b"), "topic": "t", "a": "a", "b": "b", "similarity": 0.5, "nli_ab": NLI_CON, "nli_ba": NLI_CON,
           "llm_pair": llm_output(), "llm_direct_context": {"label": "QUALIFIES", "reason": "x"}}
    data = TopicData(Topic("t", "dev", "q?", []), papers, {"a": pair.a, "b": pair.b}, [rec], threshold=0.12)
    s = Systems(data)
    labels = {name: predict(s, name, rec)[0] for name in ("similarity", "nli", "llm_direct_claims", "llm_direct_context",
                                                         "hybrid_full", "hybrid_no_gate", "hybrid_always_qualify",
                                                         "hybrid_always_abstain", "always_unrelated")}
    assert labels == {"similarity": "SUPPORTS", "nli": "CONTRADICTS", "llm_direct_claims": None, "llm_direct_context": "QUALIFIES",
                      "hybrid_full": "QUALIFIES", "hybrid_no_gate": "CONTRADICTS", "hybrid_always_qualify": "QUALIFIES",
                      "hybrid_always_abstain": M.ABSTAIN, "always_unrelated": "UNRELATED"}
    # the conflict verdict is a second level, reported next to the relation
    assert predict_full(s, "hybrid_full", rec)[:2] == ("QUALIFIES", "context_dependent")
    assert predict_full(s, "hybrid_no_gate", rec)[:2] == ("CONTRADICTS", "genuine")
    assert predict_full(s, "hybrid_always_abstain", rec)[:2] == (M.ABSTAIN, "insufficient")
    assert two_level(M.ABSTAIN, "insufficient") == ("CONTRADICTS", "insufficient")
    assert predict_full(s, "nli", rec)[:2] == ("CONTRADICTS", "genuine") and predict_full(s, "similarity", rec)[1] is None
    assert json.dumps(rec)  # records stay JSON-serialisable
    assert ann.stratum_of(labels) == "gate_input"  # the full system sees a face-value conflict here
    assert ann.stratum_of({**labels, "hybrid_no_gate": "QUALIFIES"}) == "conflict"  # only other systems flag it
    assert ann.stratum_of({**labels, "hybrid_full": M.ABSTAIN}) == "full_assert"
    assert ann.stratum_of({"similarity": "SUPPORTS", "nli": "UNRELATED"}) == "background"


def test_partial_stance_is_a_qualification_not_a_conflict(llm_pair):
    pair, papers = llm_pair
    view = view_from_output(llm_output(stance="partial", status="same"), pair.a, pair.b, papers)
    rel = clf(0.0, 0.0).decide(pair, NLI_NEU, NLI_NEU, view)
    assert rel.relation == RelationType.QUALIFIES and not rel.signals.conflict_signal
    assert final_label(rel, RuleConfig(), {"a": pair.a, "b": pair.b}) == "QUALIFIES"  # no conflict verdict is attached


def test_one_material_difference_is_enough_for_the_llm_gate(llm_pair):
    pair, papers = llm_pair
    out = llm_output()
    out["factors"] += [{"factor": f, "a_value": "ImageNet", "a_quote": "JFT-300M", "b_value": "ImageNet", "b_quote": "ImageNet",
                        "status": "same", "material": False} for f in ("dataset", "task", "evaluation metric", "baseline")]
    view = view_from_output(out, pair.a, pair.b, papers)
    rel = clf(0.0, 0.0).decide(pair, NLI_NEU, NLI_NEU, view)
    assert rel.signals.context_divergence < 0.3 and rel.relation == RelationType.QUALIFIES
