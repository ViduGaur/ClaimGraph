"""The pre-registered sampling and verdict logic of the fresh-topic E1 validation."""
from app.evaluation import fresh_validation as V


def test_allocation_takes_the_trigger_pairs_first_and_passes_on_unused_quota():
    n = V.allocate({"A": 26, "B": 27, "C": 21, "D": 81, "E": 445}, 80)
    assert n == {"A": 26, "B": 27, "C": 11, "D": 10, "E": 6} and sum(n.values()) == 80
    n = V.allocate({"A": 60, "B": 50, "C": 3, "D": 5, "E": 400}, 80)
    assert n["A"] == 60 and n["B"] == 0 and n["C"] == 3 and n["D"] == 5 and sum(n.values()) == 80
    assert sum(V.allocate({"A": 5, "B": 5, "C": 0, "D": 2, "E": 400}, 80).values()) == 80


def metric(p, r, gain=0.2, plo=0.7, glo=0.05):
    return {"precision_design": {"value": p, "ci": (plo, 0.95)}, "recall_c4": {"value": r, "ci": (0.4, 0.9)},
            "precision_gain": {"value": gain, "ci": (glo, 0.4)}}


TOPICS = {"t1": {"flagged": 8, "opposed": 9, "precision": 0.8, "recall": 0.7}, "t2": {"flagged": 6, "opposed": 6, "precision": 0.8, "recall": 0.6}}
GAIN = {"t1": 0.2, "t2": 0.1}


def test_outcomes_follow_the_registered_thresholds():
    assert V.outcome(metric(0.80, 0.72), TOPICS, GAIN)[0] == "strong"
    assert V.outcome(metric(0.80, 0.62), TOPICS, GAIN)[0] == "success"
    assert V.outcome(metric(0.70, 0.62), TOPICS, GAIN)[0] == "intermediate"
    assert V.outcome(metric(0.64, 0.80), TOPICS, GAIN)[0] == "failure"
    assert V.outcome(metric(0.90, 0.49), TOPICS, GAIN)[0] == "failure"
    assert V.outcome(metric(0.90, 0.80), TOPICS, {"t1": 0.2, "t2": -0.01})[0] == "failure"  # the gain rests on one topic
    collapsed = {**TOPICS, "t2": {"flagged": 6, "opposed": 6, "precision": 0.4, "recall": 0.6}}
    assert V.collapses(collapsed) == ["t2"] and V.outcome(metric(0.80, 0.72), collapsed, GAIN)[0] == "intermediate"
    assert V.collapses({"t": {"flagged": 4, "opposed": 4, "precision": 0.0, "recall": 0.0}}) == []  # too few pairs to call


def results(agree, vote2):
    return {"detectors": {"agree": agree, "vote2": vote2}}


def test_verdict_rule():
    ok, weak, bad = {**metric(0.8, 0.7), "outcome": "strong"}, {**metric(0.8, 0.7, plo=0.6), "outcome": "success"}, {**metric(0.5, 0.7), "outcome": "failure"}
    mid = {**metric(0.7, 0.6), "outcome": "intermediate"}
    assert V.verdict(results(ok, bad))[0] == "SUPPORTED"
    assert V.verdict(results(weak, bad))[0] == "PROMISING BUT INCONCLUSIVE"  # precision interval reaches the failure zone
    assert V.verdict(results({**metric(0.8, 0.7, glo=-0.1), "outcome": "strong"}, bad))[0] == "PROMISING BUT INCONCLUSIVE"
    assert V.verdict(results(mid, bad))[0] == "PROMISING BUT INCONCLUSIVE"
    assert V.verdict(results(bad, bad))[0] == "FALSIFIED"
