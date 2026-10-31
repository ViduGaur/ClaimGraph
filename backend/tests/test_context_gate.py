"""The context gate, pinned case by case. docs/context-gate.md describes exactly this behaviour."""
from __future__ import annotations

import pytest

from app.evaluation.experiment import final_label
from app.evaluation.metrics import ABSTAIN
from app.providers.llm import BudgetedLLM
from app.reasoning.relations import FACTORS, PairCandidate, PairView, RelationClassifier, RuleConfig, context_divergence
from app.schemas.domain import FactorComparison, Polarity, RelationType
from tests.conftest import make_claim

NEUTRAL = {"entailment": 0.02, "contradiction": 0.02, "neutral": 0.96}
CONTRA = {"entailment": 0.01, "contradiction": 0.95, "neutral": 0.04}
ENTAIL = {"entailment": 0.9, "contradiction": 0.01, "neutral": 0.09}
LABELS = [label for label, _, _ in FACTORS]


def factors(**status) -> list[FactorComparison]:
    """All twelve factors; the named ones get the given status, the rest are unknown."""
    by = {k.replace("_", " ").replace("training and design", "training & design").replace("model method", "model / method"): v
          for k, v in status.items()}
    assert set(by) <= set(LABELS), set(by) - set(LABELS)
    return [FactorComparison(factor=f, a=["x"] if f in by else [], b=["y"] if f in by else [], status=by.get(f, "unknown")) for f in LABELS]


def view(stance: str | None, source: str = "llm", **status) -> PairView:
    related = stance is not None
    return PairView(stance=stance, comparable=related if source == "llm" else None, same_direction=stance == "agree",
                    same_subject=related, shared_object=related, factors=factors(**status), confident=source == "llm", source=source)


def decide(v: PairView, nli=NEUTRAL, similarity: float = 0.75, config: RuleConfig | None = None):
    a = make_claim("a", "p1", "Method X improves accuracy on benchmark one.")
    b = make_claim("b", "p2", "Method X does not improve accuracy on benchmark two.", polarity=Polarity.NEGATIVE)
    clf = RelationClassifier(None, BudgetedLLM(None, 0), pair_threshold=0.72, config=config)  # type: ignore[arg-type]
    rel = clf.decide(PairCandidate(a, b, similarity), nli, nli, v)
    return rel, final_label(rel, config or RuleConfig(), {"a": a, "b": b})


# ---- the three outcomes of the gate -----------------------------------------------------------------
def test_no_context_stated_by_both_papers_means_insufficient_evidence():
    rel, label = decide(view("oppose"))
    assert rel.relation == RelationType.CONTRADICTS and rel.uncertain and label == ABSTAIN


def test_a_stated_material_mismatch_makes_the_conflict_context_dependent():
    rel, label = decide(view("oppose", dataset="different"))
    assert rel.relation == RelationType.QUALIFIES and rel.signals.conflict_signal and label == "QUALIFIES"


def test_comparable_context_without_a_mismatch_lets_the_contradiction_stand():
    rel, label = decide(view("oppose", dataset="same", task="overlap"))
    assert rel.relation == RelationType.CONTRADICTS and not rel.uncertain and label == "CONTRADICTS"


def test_an_incidental_difference_recorded_as_overlap_never_decides():
    assert decide(view("oppose", dataset="overlap", evaluation_metric="overlap"))[1] == "CONTRADICTS"


# ---- which factors count --------------------------------------------------------------------------------
def test_method_and_baseline_alone_are_not_experimental_context():
    # only what the claims are about is stated by both sides: the gate cannot compare conditions
    assert decide(view("oppose", model_method="different", baseline="different"))[1] == ABSTAIN


def test_method_mismatch_is_decisive_once_some_condition_is_known():
    # an asymmetry of the implementation, documented as such
    assert decide(view("oppose", task="same", model_method="different"))[1] == "QUALIFIES"


def test_one_mismatch_is_enough_for_an_llm_table_but_not_for_a_lexicon_table():
    many_same = dict(dataset="same", task="same", population="same", evaluation_metric="same", data_scale="same", sample_size="different")
    llm = view("oppose", **many_same)
    div = context_divergence(llm.factors)
    assert div < 0.3
    assert decide(llm)[1] == "QUALIFIES"  # semantic table: any stated material mismatch decides
    assert decide(view("oppose", source="rules", **many_same), nli=CONTRA)[1] == "CONTRADICTS"  # lexicon table: divergence >= 0.3 needed


# ---- weights ------------------------------------------------------------------------------------------------
def test_factor_weights_are_unequal_and_only_shape_the_divergence_score():
    weights = {label: w for label, _, w in FACTORS}
    assert weights["data scale"] == 1.5 and weights["sample size"] == 0.4 and len(set(weights.values())) > 1
    assert sum(weights.values()) == pytest.approx(10.6)
    f = factors(data_scale="different", evaluation_metric="same")
    assert context_divergence(f) == pytest.approx(1.5 / (1.5 + 0.8))  # 0.65, not the 0.5 equal weights would give
    assert context_divergence(factors(sample_size="different", data_scale="same")) == pytest.approx(0.4 / 1.9)
    # in the hybrid system both of these are context-dependent all the same: the weights do not change the verdict
    assert decide(view("oppose", data_scale="different", evaluation_metric="same"))[1] == "QUALIFIES"
    assert decide(view("oppose", sample_size="different", data_scale="same"))[1] == "QUALIFIES"


def test_paper_level_values_count_half_in_the_divergence():
    f = factors(dataset="different", task="same")
    f = [x.model_copy(update={"inherited": True}) if x.factor == "dataset" else x for x in f]
    assert context_divergence(f) == pytest.approx(0.6 / (0.6 + 1.0))


def test_unknown_factors_are_ignored_not_counted_as_different():
    assert context_divergence(factors()) is None
    assert context_divergence(factors(task="same")) == 0.0


# ---- stance and NLI ---------------------------------------------------------------------------------------
def test_the_gate_only_runs_on_a_face_value_conflict():
    assert decide(view("agree", dataset="different"))[1] == "EXTENDS"
    assert decide(view("agree", evaluation_metric="different"))[1] == "SUPPORTS"  # not one of the 'new setting' factors
    assert decide(view("agree"))[1] == "SUPPORTS"


def test_semantic_stance_overrides_nli():
    assert decide(view(None, dataset="different"), nli=CONTRA)[1] == "UNRELATED"  # 'unrelated' ends the analysis
    assert decide(view("agree"), nli=CONTRA)[1] == "SUPPORTS"  # an agreeing stance blocks the NLI contradiction
    rel, label = decide(view("oppose", dataset="different"), nli=ENTAIL)
    assert label == "QUALIFIES" and rel.signals.conflict_signal  # opposition needs no NLI corroboration


def test_nli_only_matters_for_a_partial_stance():
    rel, label = decide(view("parity_mismatch", dataset="same"))
    assert label == "QUALIFIES" and not rel.signals.conflict_signal  # narrowing, not a conflict: no verdict
    rel, label = decide(view("parity_mismatch", dataset="same"), nli=CONTRA)
    assert label == "CONTRADICTS" and rel.signals.conflict_signal  # confident NLI contradiction turns it into a conflict
    assert decide(view("parity_mismatch", dataset="different"), nli=CONTRA)[1] == "QUALIFIES"
    assert decide(view("parity_mismatch"), nli=ENTAIL)[1] == "SUPPORTS"  # confident NLI entailment turns it into agreement


def test_partial_stance_does_not_depend_on_embedding_similarity():
    assert decide(view("parity_mismatch", task="same"), similarity=0.73)[1] == "QUALIFIES"
    assert decide(view("parity_mismatch", source="rules", task="same"), similarity=0.73)[1] == "UNRELATED"  # surface cue: needs >= 0.79


# ---- ablations --------------------------------------------------------------------------------------------
def test_without_the_gate_every_face_value_conflict_is_asserted():
    off = RuleConfig(use_context_gate=False)
    for v in (view("oppose"), view("oppose", dataset="different"), view("oppose", dataset="same")):
        assert decide(v, config=off)[1] == "CONTRADICTS"


def test_without_stance_only_confident_nli_raises_a_conflict():
    off = RuleConfig(use_stance=False)
    assert decide(view("oppose", dataset="different"), config=off)[1] == "UNRELATED"
    assert decide(view("oppose", dataset="different"), nli=CONTRA, config=off)[1] == "QUALIFIES"
    assert decide(view(None, dataset="same"), nli=CONTRA, config=off)[1] == "CONTRADICTS"  # the 'unrelated' veto is part of stance
