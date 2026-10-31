"""V2 comparability model: evidence features, sufficiency, the decision and the nested V1 control."""
from __future__ import annotations

import math

import pytest

from app.reasoning import comparability as C
from app.reasoning.relations import FACTORS
from app.schemas.domain import FactorComparison


def fc(factor: str, status: str, *, inherited: bool = False, immaterial: bool = False) -> FactorComparison:
    known = status != "unknown"
    return FactorComparison(factor=factor, a=["x"] if known else [], b=["y"] if known else [], status=status, inherited=inherited,
                            note=C.IMMATERIAL_NOTE if immaterial else None)


def params(**kw) -> C.ComparabilityParams:
    base = dict(name="t", intercept=0.0, weight_same=0.0, weights_different={"scope": 1.0, "measurement": 0.5}, threshold=0.5)
    return C.ComparabilityParams(**{**base, **kw})


def test_every_factor_has_exactly_one_role():
    assert set(C.ROLES) == {label for label, _, _ in FACTORS}
    assert set(C.ROLES.values()) == set(C.ROLE_ORDER)


def test_unknown_factor_is_no_evidence_either_way():
    ev = C.factor_evidence([fc("dataset", "unknown")])
    assert all(not e.known and e.same == 0 and e.different == 0 for e in ev)
    assert C.sufficiency(ev) == 0


def test_statuses_map_to_signed_evidence():
    ev = {e.factor: e for e in C.factor_evidence([fc("dataset", "different"), fc("task", "same"), fc("evaluation metric", "overlap")])}
    assert (ev["dataset"].different, ev["dataset"].same) == (1.0, 0.0)
    assert (ev["task"].different, ev["task"].same) == (0.0, 1.0)
    assert (ev["evaluation metric"].different, ev["evaluation metric"].same) == (0.5, 0.0)


def test_paper_level_statement_counts_half():
    ev = {e.factor: e for e in C.factor_evidence([fc("dataset", "different", inherited=True)])}
    assert ev["dataset"].different == 0.5 and C.sufficiency(ev.values()) == 0.5
    full = {e.factor: e for e in C.factor_evidence([fc("dataset", "different", inherited=True)], grounding_weight=False)}
    assert full["dataset"].different == 1.0


def test_llm_materiality_flag_is_ignored_unless_asked_for():
    table = [fc("dataset", "overlap", immaterial=True)]
    assert next(e for e in C.factor_evidence(table) if e.factor == "dataset").different == 1.0
    assert next(e for e in C.factor_evidence(table, llm_materiality=True) if e.factor == "dataset").different == 0.5


def test_sufficiency_counts_only_scope_and_measurement():
    ev = C.factor_evidence([fc("model / method", "different"), fc("baseline", "same"), fc("sample size", "same")])
    assert C.sufficiency(ev) == 0
    assert C.sufficiency(C.factor_evidence([fc("dataset", "same"), fc("evaluation metric", "different")])) == 2


def test_not_opposed_has_no_conflict_status():
    assert C.assess([fc("dataset", "different")], params(), opposed=False).status == "none"


def test_no_shared_context_is_insufficient_whatever_the_score():
    r = C.assess([fc("model / method", "same")], params(intercept=5.0))
    assert r.status == "insufficient" and r.p_comparable > 0.99


def test_mismatches_accumulate_and_agreement_offsets_them():
    q = params(weight_same=1.0)
    one = C.assess([fc("dataset", "different")], q)
    two = C.assess([fc("dataset", "different"), fc("task", "different")], q)
    offset = C.assess([fc("dataset", "different"), fc("task", "same")], q)
    assert two.p_comparable < one.p_comparable < offset.p_comparable
    assert one.status == "context_dependent" and offset.status == "genuine"
    assert offset.contributions == {"dataset": -1.0, "task": 1.0}


def test_a_role_with_zero_weight_cannot_explain_a_conflict_away():
    r = C.assess([fc("dataset", "same"), fc("model / method", "different"), fc("baseline", "different")], params())
    assert r.status == "genuine" and "model / method" not in r.contributions


def test_factor_weight_overrides_role_weight():
    q = params(weights_different={"scope": 1.0, "dataset": 3.0})
    assert C.assess([fc("dataset", "different")], q).contributions == {"dataset": -3.0}
    assert C.assess([fc("task", "different")], q).contributions == {"task": -1.0}


def test_v1_gate_is_the_limit_of_heavy_mismatch_weights():
    """With a very large weight on every material mismatch and none on agreement the model reproduces the V1 gate."""
    q = params(intercept=1.0, weights_different={r: 50.0 for r in C.ROLE_ORDER}, llm_materiality=True, grounding_weight=False)
    tables = [
        [fc("dataset", "same"), fc("task", "same")],
        [fc("dataset", "different"), fc("task", "same")],
        [fc("dataset", "same"), fc("model / method", "different")],
        [fc("model / method", "different")],
        [],
    ]
    for table in tables:
        assert C.assess(table, q).status == C.v1_status(table), table


def test_v1_control_matches_the_documented_gate():
    assert C.v1_status([]) == "insufficient"
    assert C.v1_status([fc("model / method", "different")]) == "insufficient"  # the documented asymmetry
    assert C.v1_status([fc("dataset", "same"), fc("model / method", "different")]) == "context_dependent"
    assert C.v1_status([fc("dataset", "overlap", immaterial=True)]) == "genuine"


def test_typed_gate_separates_intervention_from_context():
    assert C.typed_gate_status([fc("dataset", "same"), fc("model / method", "different")]) == "none"
    assert C.typed_gate_status([fc("dataset", "different")]) == "context_dependent"
    assert C.typed_gate_status([fc("baseline", "same")]) == "insufficient"


def test_parameters_round_trip_and_a_missing_threshold_never_asserts(tmp_path):
    q = C.ComparabilityParams.from_fit("m", -1.0, {"same": 0.2, "diff:context": 0.7, "diff:intervention": 0.1}, None, {"calibrated_on": "dev"})
    assert q.weights_different == {"scope": 0.7, "measurement": 0.7, "intervention": 0.1} and math.isinf(q.threshold)
    C.save_params(tmp_path / "p.json", [q])
    back = C.load_params(tmp_path / "p.json")["m"]
    assert back == q
    assert C.assess([fc("dataset", "same")], back).status == "context_dependent"
    assert C.load_params(tmp_path / "missing.json") == {}


@pytest.mark.parametrize("status", ["same", "overlap", "different"])
def test_assess_is_deterministic(status):
    table = [fc("dataset", status), fc("evaluation metric", "different")]
    assert C.assess(table, params()) == C.assess(table, params())


def test_named_values_are_compared_as_sets_of_names():
    assert C.deterministic_status("dataset", ["CIFAR-10, CIFAR-100"], ["cifar100 and CIFAR 10"]) == "same"
    assert C.deterministic_status("dataset", ["CIFAR-10, ImageNet"], ["ImageNet"]) == "overlap"
    assert C.deterministic_status("evaluation metric", ["top-1 accuracy"], ["BLEU"]) == "different"


def test_quantities_are_compared_by_number():
    assert C.deterministic_status("sample size", ["1,000 participants"], ["1k subjects"]) == "same"
    assert C.deterministic_status("data scale", ["300M images"], ["1.2 million images"]) == "different"


def test_free_text_factors_and_missing_values_have_no_deterministic_status():
    assert C.deterministic_status("experimental conditions", ["pre-trained"], ["from scratch"]) is None
    assert C.deterministic_status("dataset", [], ["ImageNet"]) is None
    table = [fc("experimental conditions", "different"), FactorComparison(factor="dataset", a=["ImageNet"], b=["ImageNet"], status="different")]
    out = {f.factor: f.status for f in C.with_deterministic_statuses(table)}
    assert out == {"experimental conditions": "different", "dataset": "same"}
