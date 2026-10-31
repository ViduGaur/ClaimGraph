"""Conditional-prediction frames (E3): the LLM extracts one record per claim; everything after is code."""
import numpy as np

from app.evaluation import layer_study as L
from app.reasoning import frames as F
from tests.conftest import make_claim, make_paper

CLAIM = "In adults over 60, drug D lowered blood pressure more than placebo."


def raw(**kw):
    out = {"intervention": "drug D", "intervention_is_new": False, "comparator": "placebo", "outcome": "blood pressure reduction",
           "effect": "positive", "effect_quote": "drug D lowered blood pressure more than placebo", "conditional": True,
           "conditions": [{"factor": "population", "value": "adults over 60", "quote": "In adults over 60"}]}
    return {**out, **kw}


def frame(cid="c1", **kw):
    return F.frame_from_output(raw(**kw), make_claim(cid, "p1", CLAIM), make_paper("p1"))


def sim(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def test_prompt_never_asks_for_a_decision_and_sees_one_claim():
    text = F.SYSTEM.lower()
    assert "do not compare it with any other paper" in text
    for word in ("contradict", "genuine", "context-dependent", "insufficient", "supports", "qualifies"):
        assert word not in text
    assert set(F.SCHEMA["properties"]) == {"intervention", "intervention_is_new", "comparator", "outcome", "effect", "effect_quote", "conditional", "conditions"}
    assert "model / method" not in F.CONDITION_FACTORS and len(F.CONDITION_FACTORS) == 10
    rendered = F.render_claim(make_claim("c1", "p1", CLAIM), make_paper("p1"))
    assert rendered.count("CLAIM:") == 1 and "CLAIM B" not in rendered


def test_quotes_are_checked_in_code():
    fr = frame()
    assert fr.states_effect and fr.conditions["population"].level == "claim"
    assert not frame(effect_quote="drug D is clearly superior in every respect").states_effect  # not in the claim sentence
    assert not frame(effect="not_stated", effect_quote=None).states_effect
    dropped = frame(conditions=[{"factor": "dataset", "value": "ImageNet", "quote": "evaluated on ImageNet"}])
    assert dropped.conditions == {}  # a value whose quote is not in the paper is dropped
    assert frame(conditions=[{"factor": "model / method", "value": "x", "quote": "drug D"}]).conditions == {}
    assert F.frame_from_output({"effect": "decides"}, make_claim("c1", "p1", CLAIM), make_paper("p1")) is None


def test_new_method_must_be_named_by_the_other_claim():
    assert not F.names_match("normalized direction-preserving Adam (ND-Adam)", "Adam-like adaptive algorithms")
    assert F.names_match("sharpness-aware minimization (SAM)", "sharpness-aware minimization")
    assert F.names_match("SAM", "the SAM optimizer")


def test_effects_and_role_swap():
    assert F.effect_relation("positive", "negative") == "opposite"
    assert F.effect_relation("positive", "none") == "effect_vs_none"
    assert F.effect_relation("mixed", "positive") == "mixed"
    a = frame(intervention="adam", comparator="sgd", effect="negative", effect_quote="lowered blood pressure")
    b = frame("c2", intervention="sgd", comparator="adam", effect="positive", effect_quote="lowered blood pressure")
    swapped = F.compare_frames(a, b, sim)
    assert swapped.alignment != "direct" and swapped.effects == "agree"  # 'Adam worse than SGD' = 'SGD better than Adam'
    assert F.compare_frames(a, b, sim, allow_swap=False).alignment == "direct"


def test_relation_and_status_are_decided_in_code():
    a = frame()
    b = frame("c2", effect="none", effect_quote="lowered blood pressure", conditions=[{"factor": "population", "value": "children", "quote": "In adults over 60"}])
    c = F.compare_frames(a, b, sim)
    assert c.incompatible and c.shared == {"population": "different"}
    assert F.frame_relation(c, 0.5) == "CONTRADICTS" and F.frame_status(c, 0.5) == "context_dependent"
    assert F.frame_relation(c, 1.1) == "UNRELATED" and F.frame_status(c, 1.1) == "none"
    same = F.compare_frames(frame(conditional=False), frame("c2", effect="negative", effect_quote="lowered blood pressure", conditional=False), sim)
    assert F.frame_status(same, 0.5) == "genuine"
    bare = F.compare_frames(frame(conditional=False, conditions=[]), frame("c2", effect="negative", effect_quote="lowered blood pressure", conditional=False, conditions=[]), sim)
    assert F.frame_status(bare, 0.5) == "insufficient"
    silent = F.compare_frames(a, frame("c2", effect="not_stated", effect_quote=None), sim)
    assert F.frame_relation(silent, 0.0) == "UNRELATED"
    assert F.condition_status("dataset", "CIFAR-10", "ImageNet") == "different"
    assert F.condition_status("population", "small models", "small models only") == "same"


def test_study_statistics():
    boot = L.Boot(["a", "a", "b", "c"], 50, 0)
    assert all(len(s) >= 3 for s in boot.samples) and all(({0, 1} <= set(s)) or not ({0, 1} & set(s)) for s in boot.samples)
    y, pred = np.array([True, True, False, False]), np.array([True, False, True, False])
    assert L.prf(y, pred, np.ones(4)) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert L.prf(y, pred, np.array([1.0, 1.0, 3.0, 1.0]))["precision"] == 0.25  # a subsampled stratum counts for more
