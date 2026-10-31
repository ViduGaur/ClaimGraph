"""Metrics for comparing systems against gold labels, with bootstrap confidence intervals.

Two levels, kept separate
-------------------------
* **relation** — SUPPORTS, EXTENDS, QUALIFIES, CONTRADICTS, or UNRELATED when no relation holds.
  Gold labels are always one of these five.
* **conflict verdict** — only for pairs that conflict at face value: ``genuine``,
  ``context_dependent`` or ``insufficient``.

A system prediction is one of the five relations or :data:`ABSTAIN`. ABSTAIN is what a system
outputs when it sees a face-value conflict but declines to assert it: relation CONTRADICTS flagged
uncertain, verdict ``insufficient``. In the relation-level scores an abstention is never correct
and is never an asserted contradiction; :func:`remap_abstentions` gives the two alternative
readings (count it as CONTRADICTS, or as QUALIFIES) for sensitivity checks.

Quantities
----------
* **false contradictions per 100 pairs** — pairs wrongly asserted as CONTRADICTS, per 100 pairs.
* **false-contradiction rate (FCR)** — of the asserted contradictions, the share that are not
  gold contradictions (= 1 − contradiction precision); undefined when none is asserted.
* **contradiction precision / recall / F1**, **per-class precision / recall / F1**,
  **macro-F1** over the five relations, **accuracy**.
* **verdict accuracy** — on gold face-value conflicts (gold verdict known): is the system's
  verdict the gold one? A missed conflict counts as wrong; ``verdict_acc_detected`` conditions on
  the system having flagged a conflict.

Every example carries a weight. With weight 1 the scores describe the evaluation sample; with
design weights (stratum size / sampled items) they estimate the same quantity over the whole
candidate pool the sample was drawn from.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace

RELATIONS = ["SUPPORTS", "EXTENDS", "QUALIFIES", "CONTRADICTS", "UNRELATED"]
ABSTAIN = "ABSTAIN"
PRED_LABELS = [*RELATIONS, ABSTAIN]
VERDICTS = ["genuine", "context_dependent", "insufficient"]


@dataclass(frozen=True)
class Example:
    item_id: str
    topic: str
    gold: str  # one of RELATIONS
    pred: str | None  # one of PRED_LABELS; None = the system produced no output for this pair
    gold_verdict: str | None = None  # set when the pair is a gold face-value conflict
    pred_verdict: str | None = None  # set when the system treated the pair as a face-value conflict
    weight: float = 1.0
    stratum: str = ""


def _pred(e: Example) -> str:
    return e.pred or "UNRELATED"  # no output = nothing asserted


def unweighted(ex: list[Example]) -> list[Example]:
    return [replace(e, weight=1.0) for e in ex]


def remap_abstentions(ex: list[Example], to: str) -> list[Example]:
    """Sensitivity reading: score every abstention as if the system had said ``to``."""
    return [replace(e, pred=to) if e.pred == ABSTAIN else e for e in ex]


def _w(ex, cond) -> float:
    return sum(e.weight for e in ex if cond(e))


def prf(ex: list[Example], is_gold: Callable[[Example], bool], is_pred: Callable[[Example], bool]) -> dict:
    tp = _w(ex, lambda e: is_gold(e) and is_pred(e))
    fp = _w(ex, lambda e: not is_gold(e) and is_pred(e))
    fn = _w(ex, lambda e: is_gold(e) and not is_pred(e))
    return {"p": tp / (tp + fp) if tp + fp else None, "r": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None, "tp": tp, "fp": fp, "fn": fn}


def class_prf(ex: list[Example], label: str) -> dict:
    return prf(ex, lambda e: e.gold == label, lambda e: _pred(e) == label)


def macro_f1(ex: list[Example]) -> float | None:
    """Mean F1 over the relations that occur in the gold labels or the predictions."""
    scores = []
    for c in RELATIONS:
        s = class_prf(ex, c)
        if s["tp"] + s["fp"] + s["fn"] == 0:
            continue  # class absent on both sides: undefined, not zero
        scores.append(s["f1"] or 0.0)
    return sum(scores) / len(scores) if scores else None


def _total(ex) -> float:
    return sum(e.weight for e in ex)


def _fcr(ex: list[Example]) -> float | None:
    asserted = _w(ex, lambda e: _pred(e) == "CONTRADICTS")
    if not asserted:
        return None  # no contradiction asserted: the rate is undefined (see n_pred_con)
    return _w(ex, lambda e: _pred(e) == "CONTRADICTS" and e.gold != "CONTRADICTS") / asserted


def _verdict(ex: list[Example], *, detected_only: bool) -> float | None:
    pool = [e for e in ex if e.gold_verdict and (not detected_only or e.pred_verdict)]
    return _w(pool, lambda e: e.pred_verdict == e.gold_verdict) / _total(pool) if pool else None


METRIC_FNS: dict[str, Callable[[list[Example]], float | None]] = {
    "accuracy": lambda ex: _w(ex, lambda e: _pred(e) == e.gold) / _total(ex) if ex else None,
    "macro_f1": macro_f1,
    "con_precision": lambda ex: class_prf(ex, "CONTRADICTS")["p"],
    "con_recall": lambda ex: class_prf(ex, "CONTRADICTS")["r"],
    "con_f1": lambda ex: class_prf(ex, "CONTRADICTS")["f1"],
    "false_con_per_100": lambda ex: _w(ex, lambda e: _pred(e) == "CONTRADICTS" and e.gold != "CONTRADICTS") / _total(ex) if ex else None,
    "fcr": _fcr,
    "qual_f1": lambda ex: class_prf(ex, "QUALIFIES")["f1"],
    "conflict_recall": lambda ex: prf(ex, lambda e: bool(e.gold_verdict), lambda e: bool(e.pred_verdict))["r"],
    "conflict_f1": lambda ex: prf(ex, lambda e: bool(e.gold_verdict), lambda e: bool(e.pred_verdict))["f1"],
    "verdict_acc": lambda ex: _verdict(ex, detected_only=False),
    "verdict_acc_detected": lambda ex: _verdict(ex, detected_only=True),
    "abstention": lambda ex: _w(ex, lambda e: _pred(e) == ABSTAIN) / _total(ex) if ex else None,
}
LOWER_IS_BETTER = {"false_con_per_100", "fcr"}


def counts(ex: list[Example]) -> dict:
    """Plain item counts (never weighted)."""
    return {"n": len(ex), "coverage": sum(1 for e in ex if e.pred) / len(ex) if ex else None,
            "n_gold_con": sum(1 for e in ex if e.gold == "CONTRADICTS"),
            "n_gold_conflict": sum(1 for e in ex if e.gold_verdict),
            "n_pred_con": sum(1 for e in ex if _pred(e) == "CONTRADICTS"),
            "n_false_con": sum(1 for e in ex if _pred(e) == "CONTRADICTS" and e.gold != "CONTRADICTS"),
            "n_abstain": sum(1 for e in ex if _pred(e) == ABSTAIN)}


def _resample(ex: list[Example], rng: random.Random) -> list[int]:
    """Indices of one bootstrap draw; items are resampled within their sampling stratum."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(ex):
        groups[e.stratum].append(i)
    idx: list[int] = []
    for members in groups.values():
        idx += [members[rng.randrange(len(members))] for _ in members]
    return idx


def bootstrap(ex: list[Example], fn: Callable[[list[Example]], float | None], *, n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile bootstrap over items. Draws where the metric is undefined are skipped."""
    if not ex:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        v = fn([ex[i] for i in _resample(ex, rng)])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None, None  # too unstable to report an interval
    vals.sort()
    return vals[int(alpha / 2 * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]


def paired_bootstrap(a: list[Example], b: list[Example], fn: Callable[[list[Example]], float | None], *, n_boot: int = 2000,
                     seed: int = 0) -> dict | None:
    """Difference (a − b) of a metric on the same items, with a 95% interval and a two-sided bootstrap p-value."""
    assert [e.item_id for e in a] == [e.item_id for e in b], "systems must be scored on the same items in the same order"
    va, vb = fn(a), fn(b)
    if va is None or vb is None:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample(a, rng)
        da, db = fn([a[i] for i in idx]), fn([b[i] for i in idx])
        if da is not None and db is not None:
            diffs.append(da - db)
    if len(diffs) < n_boot * 0.5:
        return {"delta": va - vb, "lo": None, "hi": None, "p": None}
    diffs.sort()
    le = sum(1 for d in diffs if d <= 0) / len(diffs)
    ge = sum(1 for d in diffs if d >= 0) / len(diffs)
    return {"delta": va - vb, "lo": diffs[int(0.025 * len(diffs))], "hi": diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))],
            "p": min(1.0, 2 * min(le, ge))}


def evaluate(ex: list[Example], *, n_boot: int = 2000, seed: int = 0) -> dict:
    out: dict = {"counts": counts(ex), "metrics": {}, "per_class": {}}
    for name, fn in METRIC_FNS.items():
        value = fn(ex)
        lo, hi = bootstrap(ex, fn, n_boot=n_boot, seed=seed) if n_boot else (None, None)
        out["metrics"][name] = {"value": value, "lo": lo, "hi": hi}
    for label in RELATIONS:
        s = class_prf(ex, label)
        out["per_class"][label] = {"precision": s["p"], "recall": s["r"], "f1": s["f1"],
                                   "support": sum(1 for e in ex if e.gold == label),
                                   "predicted": sum(1 for e in ex if _pred(e) == label)}
    return out


def confusion(ex: list[Example]) -> dict[str, dict[str, int]]:
    """Rows = gold relation, columns = predicted relation or ABSTAIN (item counts)."""
    table = {g: {p: 0 for p in PRED_LABELS} for g in RELATIONS}
    for e in ex:
        table[e.gold][_pred(e)] += 1
    return table


def label_distribution(labels: list[str]) -> dict[str, int]:
    return {k: sum(1 for x in labels if x == k) for k in RELATIONS}
