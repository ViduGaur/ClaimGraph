"""Weight-of-evidence comparability model (the V2 decision layer).

V1 collapsed three questions into one gate: any one material context mismatch turned a face-value conflict into
"context-dependent". V2 keeps them apart:

A. opposition     do the claims oppose at face value?            (semantic stance, as in V1)
B. comparability  were they made under comparable conditions?    (typed, signed factor evidence)
C. sufficiency    is enough shared context stated to judge B?    (a count)

The factor table comes from a frozen LLM with its quotes checked in code; factor roles, evidence features and the
sufficiency count are deterministic. Only the coefficients and decision threshold of B are fitted, on development data
(``ComparabilityParams``). The V1 gate is the limiting case "any mismatch weighs infinitely, agreement nothing" and is
kept as the control (``v1_status``).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from app.schemas.domain import FactorComparison

# Role of each factor. A hypothesis about the factors, tested in the development study, not an assumption.
ROLES: dict[str, str] = {
    "dataset": "scope", "data scale": "scope", "population": "scope", "task": "scope",
    "experimental conditions": "scope", "training & design": "scope",
    "evaluation metric": "measurement", "methodology": "measurement",
    "sample size": "strength", "time period": "strength",
    "model / method": "intervention", "baseline": "intervention",
}
ROLE_ORDER = ["scope", "measurement", "strength", "intervention"]
FACTOR_ORDER = list(ROLES)
SUFFICIENCY_ROLES = ("scope", "measurement")  # what counts as shared experimental context for question C
IMMATERIAL_NOTE = "stated difference judged incidental to the findings"  # set by alignment.view_from_output

STATUSES = ("none", "genuine", "context_dependent", "insufficient")


@dataclass(frozen=True)
class FactorEvidence:
    factor: str
    role: str
    known: bool  # stated by both papers with a grounded quote
    same: float  # e_k: evidence that the settings are comparable on this factor
    different: float  # d_k: evidence that they are not
    grounding: float  # 1 claim-level, 0.5 paper-level


def factor_evidence(factors: Iterable[FactorComparison], *, llm_materiality: bool = False,
                    grounding_weight: bool = True) -> list[FactorEvidence]:
    """Signed evidence per factor.

    ``d_k = g_k * (1[different] + 0.5 * 1[partly the same])`` and ``e_k = g_k * 1[same]``; a factor that is not stated
    by both papers contributes nothing either way. Without ``llm_materiality`` a difference the LLM called
    immaterial still counts as a difference: whether a kind of difference matters is the job of the calibrated
    weight of its role, not of an unchecked per-pair judgement.
    """
    by_name = {f.factor: f for f in factors}
    out = []
    for name in FACTOR_ORDER:
        f = by_name.get(name)
        if f is None or f.status == "unknown":
            out.append(FactorEvidence(name, ROLES[name], False, 0.0, 0.0, 0.0))
            continue
        g = 0.5 if (grounding_weight and f.inherited) else 1.0
        immaterial = f.status == "overlap" and f.note == IMMATERIAL_NOTE
        if f.status == "different" or (immaterial and not llm_materiality):
            d, e = 1.0, 0.0
        elif f.status == "overlap":
            d, e = 0.5, 0.0
        else:
            d, e = 0.0, 1.0
        out.append(FactorEvidence(name, ROLES[name], True, g * e, g * d, g))
    return out


def sufficiency(evidence: Iterable[FactorEvidence]) -> float:
    """E: how many experimental-context factors both papers state (paper-level statements count half)."""
    return sum(ev.grounding for ev in evidence if ev.known and ev.role in SUFFICIENCY_ROLES)


@dataclass(frozen=True)
class ComparabilityParams:
    """Calibrated parameters of the comparability model. ``weights_different`` maps a role or a factor to its weight."""
    name: str
    intercept: float
    weight_same: float
    weights_different: dict[str, float]
    threshold: float  # assert a genuine conflict when p(comparable) >= threshold
    e_min: float = 0.5  # set a priori: at least one context factor stated by both papers, even at paper level
    llm_materiality: bool = False
    grounding_weight: bool = True
    provenance: dict = field(default_factory=dict)  # how and on what the numbers were obtained

    def as_dict(self) -> dict:
        d = asdict(self)
        d["threshold"] = None if math.isinf(self.threshold) else self.threshold  # None: nothing is ever asserted
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ComparabilityParams":
        return cls(**{**d, "threshold": math.inf if d.get("threshold") is None else d["threshold"]})

    @classmethod
    def from_fit(cls, name: str, intercept: float, columns: dict[str, float], threshold: float | None,
                 provenance: dict | None = None, **kw) -> "ComparabilityParams":
        """Parameters from a fitted model whose columns are ``same`` and ``diff:<role | factor | context | all>``."""
        weights: dict[str, float] = {}
        for col, w in columns.items():
            if not col.startswith("diff:"):
                continue
            target = col.split(":", 1)[1]
            for key in {"context": SUFFICIENCY_ROLES, "all": tuple(ROLE_ORDER)}.get(target, (target,)):
                weights[key] = w
        return cls(name, intercept, columns.get("same", 0.0), weights, math.inf if threshold is None else threshold,
                   provenance=provenance or {}, **kw)


def save_params(path: Path, params: list[ComparabilityParams], note: str = "") -> None:
    path.write_text(json.dumps({"note": note, "models": [q.as_dict() for q in params]}, indent=1))


def load_params(path: Path) -> dict[str, ComparabilityParams]:
    """Frozen parameter sets by name; empty when nothing has been calibrated yet."""
    if not path.exists():
        return {}
    return {d["name"]: ComparabilityParams.from_dict(d) for d in json.loads(path.read_text())["models"]}


@dataclass(frozen=True)
class ComparabilityResult:
    status: str  # one of STATUSES
    p_comparable: float | None
    sufficiency: float
    contributions: dict[str, float]  # factor -> signed contribution to the log-odds of "comparable"


def log_odds(evidence: list[FactorEvidence], params: ComparabilityParams) -> tuple[float, dict[str, float]]:
    contributions: dict[str, float] = {}
    for ev in evidence:
        if not ev.known:
            continue
        w = params.weights_different.get(ev.factor, params.weights_different.get(ev.role, 0.0))
        c = params.weight_same * ev.same - w * ev.different
        if c:
            contributions[ev.factor] = round(c, 4)
    return params.intercept + sum(contributions.values()), contributions


def assess(factors: Iterable[FactorComparison], params: ComparabilityParams, *, opposed: bool = True) -> ComparabilityResult:
    """Conflict status of one pair from its factor table. Deterministic given the table and ``params``."""
    evidence = factor_evidence(factors, llm_materiality=params.llm_materiality, grounding_weight=params.grounding_weight)
    e = sufficiency(evidence)
    if not opposed:
        return ComparabilityResult("none", None, e, {})
    z, contributions = log_odds(evidence, params)
    p = 1.0 / (1.0 + math.exp(-z))
    if e < params.e_min:
        status = "insufficient"
    else:
        status = "genuine" if p >= params.threshold else "context_dependent"
    return ComparabilityResult(status, round(p, 4), e, contributions)


def v1_status(factors: Iterable[FactorComparison]) -> str:
    """The V1 hard gate on a face-value conflict (LLM-read table): the nested control."""
    factors = list(factors)
    known = [f for f in factors if f.status != "unknown" and ROLES[f.factor] != "intervention"]
    if not known:
        return "insufficient"
    return "context_dependent" if any(f.status == "different" for f in factors) else "genuine"


def typed_gate_status(factors: Iterable[FactorComparison]) -> str:
    """V1 with factor roles and nothing else: an intervention mismatch means the claims are not about the same
    thing; a material scope or measurement mismatch decides; no shared context means insufficient."""
    factors = list(factors)
    if any(f.status == "different" and ROLES[f.factor] == "intervention" for f in factors):
        return "none"
    context = [f for f in factors if ROLES[f.factor] in SUFFICIENCY_ROLES]
    if not [f for f in context if f.status != "unknown"]:
        return "insufficient"
    return "context_dependent" if any(f.status == "different" for f in context) else "genuine"


# ---------------------------------------------------------------------------------------------------------------
# Deterministic comparison of stated values, where the factor allows it. Names (datasets, metrics, methods,
# baselines) are compared as sets of normalised names; quantities by their numbers. Free-text factors
# (experimental conditions, training & design, population, task, methodology) have no deterministic comparison and
# keep the status the LLM assigned.
NAMED_FACTORS = {"dataset", "evaluation metric", "model / method", "baseline"}
NUMERIC_FACTORS = {"sample size", "data scale", "time period"}
_SPLIT = re.compile(r"\s*(?:,|;|/|\band\b|\bor\b|\bvs\.?\b|\+)\s*", re.I)
_NUM = re.compile(r"(\d+(?:[.,]\d+)*)\s*(k|m|b|thousand|million|billion)?\b", re.I)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def _names(values: list[str]) -> set[str]:
    out = set()
    for v in values:
        for part in _SPLIT.split(re.sub(r"\([^)]*\)", " ", v)):
            name = re.sub(r"[^a-z0-9]+", "", part.lower())
            if name:
                out.add(name)
    return out


def _numbers(values: list[str]) -> set[float]:
    out = set()
    for v in values:
        for num, unit in _NUM.findall(v):
            out.add(float(num.replace(",", "")) * _SCALE.get(unit.lower(), 1.0))
    return out


def deterministic_status(factor: str, a: list[str], b: list[str]) -> str | None:
    """'same' | 'overlap' | 'different' from the stated values alone, or None when this factor (or these values)
    cannot be compared without reading for meaning."""
    if not a or not b:
        return None
    if factor in NAMED_FACTORS:
        sa, sb = _names(a), _names(b)
    elif factor in NUMERIC_FACTORS:
        sa, sb = _numbers(a), _numbers(b)  # type: ignore[assignment]
    else:
        return None
    if not sa or not sb:
        return None
    if sa == sb:
        return "same"
    return "overlap" if sa & sb else "different"


def with_deterministic_statuses(factors: Iterable[FactorComparison]) -> list[FactorComparison]:
    """The same table with the status of name- and number-valued factors recomputed in code."""
    out = []
    for f in factors:
        det = deterministic_status(f.factor, f.a, f.b) if f.status != "unknown" else None
        out.append(f.model_copy(update={"status": det, "note": None}) if det else f)
    return out
