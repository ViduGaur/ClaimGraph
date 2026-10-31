"""Four-state annotation protocol: parsing, the status implied by A-C, consensus, and blinding."""
from __future__ import annotations

import json

import pytest

from app.evaluation import conflict_annotation as CA
from app.evaluation.model_annotation import FORBIDDEN_IN_PROMPT
from app.providers.llm.base import LLMError

GOOD = {"stance": "oppose", "sufficiency": "sufficient", "comparability": "context_dependent", "status": "context_dependent",
        "factors": ["Dataset", "not a factor", "evaluation metric"], "justification": "x"}


def test_parse_normalises_and_keeps_only_known_factors():
    out = CA.parse(GOOD)
    assert out["status"] == "CONTEXT_DEPENDENT" and out["factors"] == ["dataset", "evaluation metric"]
    assert CA.parse({**GOOD, "comparability": "context-dependent"})["comparability"] == "context_dependent"


@pytest.mark.parametrize("bad", [{"stance": "maybe"}, {"sufficiency": "partly"}, {"status": "QUALIFIES"}, {"factors": "dataset"}])
def test_parse_rejects_answers_outside_the_protocol(bad):
    with pytest.raises(LLMError):
        CA.parse({**GOOD, **bad})


def test_status_implied_by_the_first_three_answers():
    base = {"stance": "oppose", "sufficiency": "sufficient", "comparability": "comparable"}
    assert CA.derived_status(base) == "GENUINE"
    assert CA.derived_status({**base, "comparability": "context_dependent"}) == "CONTEXT_DEPENDENT"
    assert CA.derived_status({**base, "sufficiency": "insufficient", "comparability": "not_applicable"}) == "INSUFFICIENT_EVIDENCE"
    assert CA.derived_status({**base, "stance": "not_oppose"}) == "NONE"
    assert CA.derived_status({**base, "stance": "unclear"}) is None
    assert CA.derived_status({**base, "comparability": "not_applicable"}) is None


def test_majority():
    assert CA.majority(["a", "a", "a"]) == ("a", "unanimous")
    assert CA.majority(["a", "b", "a"]) == ("a", "majority")
    assert CA.majority(["a", "b", "c"]) == (None, "split")


def test_instructions_are_zero_shot_and_blind():
    low = CA.SYSTEM.lower()
    assert not [w for w in FORBIDDEN_IN_PROMPT if w.lower() in low]
    assert "claim a:" not in low and "worked example" not in low  # definitions only, no labelled pair
    for name in CA.FACTOR_NAMES:
        assert f"- {name}:" in CA.SYSTEM
    for state in CA.STATUSES:
        assert state in CA.SYSTEM


def test_reference_keeps_dimensions_separate(tmp_path):
    def row(annotator, **kw):
        base = {"annotator": annotator, "item_id": "i1", "stance": "oppose", "sufficiency": "sufficient", "comparability": "context_dependent",
                "status": "CONTEXT_DEPENDENT", "factors": ["dataset"], "justification": ""}
        return json.dumps({**base, **kw})

    (tmp_path / "model_calls.jsonl").write_text("\n".join([
        row("gemini"), row("qwen", factors=["dataset", "task"]),
        row("haiku", sufficiency="insufficient", comparability="not_applicable", status="INSUFFICIENT_EVIDENCE", factors=[])]))
    doc = CA.build_reference(tmp_path, [{"item_id": "i1", "pair_id": "p1", "topic": "t", "run": "main", "stratum": "system_opposed"},
                                        {"item_id": "i2", "pair_id": "p2", "topic": "t", "run": "main", "stratum": "system_opposed"}])
    assert list(doc["items"]) == ["i1"]  # an item without all three annotators is left out
    r = doc["items"]["i1"]
    assert (r["stance"], r["stance_agreement"]) == ("oppose", "unanimous")
    assert (r["sufficiency"], r["sufficiency_agreement"]) == ("sufficient", "majority")
    assert r["comparability"] == "context_dependent" and r["status"] == "CONTEXT_DEPENDENT" and r["factors"] == ["dataset"]
    ag = CA.agreement(doc)
    assert ag["status"]["majority"] == 1 and ag["status_matches_derived"]["haiku"]["same"] == 1
