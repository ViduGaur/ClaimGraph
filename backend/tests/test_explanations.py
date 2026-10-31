"""Grounded conflict explanations: the structured record comes from the decision; the LLM's prose is checked."""
from __future__ import annotations

from app.providers.llm import BudgetedLLM, LLMError
from app.reasoning.conflicts import analyze_conflicts
from app.reasoning.explanations import check_explanation, evidence_record, explain_conflicts, explain_one, render_evidence
from app.reasoning.relations import PairCandidate, RelationClassifier
from app.schemas.domain import ConflictVerdict, Polarity
from tests.conftest import make_claim
from tests.test_hybrid_and_harness import ScriptedLLM, llm_output
from app.reasoning.alignment import view_from_output
from tests.conftest import make_paper

NEUTRAL = {"entailment": 0.02, "contradiction": 0.02, "neutral": 0.96}


def conflict(**factor):
    a = make_claim("a", "p1", "ViT beats CNNs when pre-trained on JFT-300M.", subject=None, polarity=Polarity.NONE)
    b = make_claim("b", "p2", "ViT is worse than CNNs when trained from scratch on ImageNet.", subject=None, polarity=Polarity.NONE)
    papers = {"p1": make_paper("p1"), "p2": make_paper("p2")}
    view = view_from_output(llm_output(**factor), a, b, papers)
    rel = RelationClassifier(None, BudgetedLLM(None, 0), pair_threshold=0.72).decide(PairCandidate(a, b, 0.8), NEUTRAL, NEUTRAL, view)  # type: ignore[arg-type]
    return analyze_conflicts([rel], {"a": a, "b": b})[0], rel, a, b


GOOD = {"explanation": "The claims oppose at face value, but the experimental conditions differ: A was pre-trained on JFT, B was "
                       "trained from scratch. The findings were not obtained under equivalent conditions, so they do not "
                       "establish a direct contradiction.", "factors_cited": ["experimental conditions"]}


def test_structured_record_comes_from_the_factor_table_not_from_the_llm():
    c, rel, a, b = conflict()
    record = evidence_record(c, rel, a, b)
    assert record["relation"] == "QUALIFIES" and record["conflict_verdict"] == "context_dependent"
    (row,) = record["decisive_mismatches"]
    assert row == {"factor": "experimental conditions", "a_value": "pre-trained on JFT", "b_value": "from scratch",
                   "a_quote": "pre-trained on JFT-300M", "b_quote": "trained from scratch on ImageNet", "paper_level": False}
    assert "A concludes" in record["apparent_conflict"]
    text = render_evidence(record)
    assert "pre-trained on JFT-300M" in text and "context_dependent" in text


def test_grounded_explanation_is_accepted():
    c, rel, a, b = conflict()
    done = explain_one(c, rel, a, b, BudgetedLLM(ScriptedLLM([GOOD]), 3))
    assert done.explanation_source == "llm" and done.llm_explanation.startswith("The claims oppose")
    assert done.verdict == c.verdict and done.differing_factors == c.differing_factors  # the decision is untouched


def test_explanations_that_go_beyond_the_evidence_are_rejected():
    c, rel, a, b = conflict()
    record = evidence_record(c, rel, a, b)
    bad = [
        ({**GOOD, "factors_cited": ["dataset"]}, "not in the table"),
        ({**GOOD, "factors_cited": []}, "none of the decisive"),
        ({"explanation": GOOD["explanation"] + " The population also differs.", "factors_cited": ["experimental conditions"]}, "population"),
        ({"explanation": GOOD["explanation"] + " Accuracy dropped by 12.5 points.", "factors_cited": ["experimental conditions"]}, "numbers"),
        ({"explanation": "Different.", "factors_cited": ["experimental conditions"]}, "length"),
        ({"explanation": GOOD["explanation"].replace("The claims oppose", "As the table shows, the claims oppose"),
          "factors_cited": ["experimental conditions"]}, "table"),
        ({"explanation": "The claims oppose at face value, but the experimental conditions of the two studies are not the same, so "
                         "the findings do not establish a direct contradiction.", "factors_cited": ["experimental conditions"]}, "values"),
        (None, "no usable response"),
    ]
    for out, why in bad:
        text, problem = check_explanation(out, record)
        assert text is None and why in problem, (out, problem)
    assert check_explanation(GOOD, record) == (" ".join(GOOD["explanation"].split()), None)
    # numbers that are in the evidence are fine
    ok = {"explanation": GOOD["explanation"].replace("pre-trained on JFT", "pre-trained on JFT-300M"), "factors_cited": ["experimental conditions"]}
    assert check_explanation(ok, record)[1] is None


def test_rejected_or_failed_explanation_falls_back_to_the_template():
    c, rel, a, b = conflict()
    bad = {**GOOD, "factors_cited": ["dataset"]}
    provider = ScriptedLLM([bad, bad])
    rejected = explain_one(c, rel, a, b, BudgetedLLM(provider, 3))
    assert rejected.explanation_source == "template" and rejected.llm_explanation is None and "rejected" in rejected.explanation_note
    assert len(provider.prompts) == 2 and "was rejected" in provider.prompts[1]  # one retry, told what was wrong
    retried = explain_one(c, rel, a, b, BudgetedLLM(ScriptedLLM([bad, GOOD]), 3))
    assert retried.explanation_source == "llm" and retried.explanation_note is None
    assert rejected.apparent_conflict and rejected.explanation == c.explanation
    failed = explain_one(c, rel, a, b, BudgetedLLM(ScriptedLLM([LLMError("down")]), 3))
    assert failed.explanation_source == "template" and failed.apparent_conflict
    no_llm = explain_conflicts([c], [rel], {"a": a, "b": b}, BudgetedLLM(None, 0))
    assert no_llm[0].explanation_source == "template" and no_llm[0].apparent_conflict


def test_insufficient_and_genuine_verdicts_get_their_own_evidence():
    c, rel, a, b = conflict(b_value=None, b_quote=None)  # one-sided → nothing comparable
    record = evidence_record(c, rel, a, b)
    assert c.verdict == ConflictVerdict.INSUFFICIENT_EVIDENCE and record["relation"] == "CONTRADICTS"
    assert not record["decisive_mismatches"] and record["one_sided_factors"][0]["stated_by"] == "A"
    c, rel, a, b = conflict(status="same")
    record = evidence_record(c, rel, a, b)
    assert c.verdict == ConflictVerdict.GENUINE and record["comparable_factors"] and not record["decisive_mismatches"]
    ok = {"explanation": "The claims oppose at face value and the experimental conditions are comparable. No stated difference "
                         "in conditions was found, so the contradiction stands on the stated evidence.", "factors_cited": ["experimental conditions"]}
    assert check_explanation(ok, record)[1] is None


def test_limit_explains_only_the_first_conflicts():
    c, rel, a, b = conflict()
    provider = ScriptedLLM([GOOD])
    out = explain_conflicts([c, c], [rel], {"a": a, "b": b}, BudgetedLLM(provider, 5), limit=1, concurrency=1)
    assert [x.explanation_source for x in out] == ["llm", "template"] and len(provider.prompts) == 1
