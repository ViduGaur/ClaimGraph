"""Contextual disagreement analysis.

Every CONTRADICTS edge, and every QUALIFIES edge that came from opposing
claims, gets a structured ConflictAnalysis. It separates:

* *context_dependent* — the claims disagree but were made under different
  data scale / datasets / conditions / training, so they may both be right;
* *genuine_disagreement* — comparable conditions, opposing findings;
* *insufficient_evidence* — the text does not state enough context to decide.
"""
from __future__ import annotations

import hashlib

from app.schemas.domain import (
    Claim,
    ConflictAnalysis,
    ConflictVerdict,
    FactorComparison,
    RelationType,
    Relationship,
)

_PRIORITY = ["data scale", "dataset", "experimental conditions", "training & design", "task", "population",
             "evaluation metric", "sample size", "methodology", "time period", "model / method", "baseline"]


def _fmt(values: list[str]) -> str:
    return ", ".join(values) if values else "unstated"


def _is_conflict(rel: Relationship) -> bool:
    # QUALIFIES is also used when one claim merely narrows the other; only a face-value disagreement is a conflict
    return rel.relation == RelationType.CONTRADICTS or (rel.relation == RelationType.QUALIFIES and rel.signals.conflict_signal)


def _factor_sentence(f: FactorComparison, a: Claim, b: Claim) -> str:
    return f"{f.factor}: {a.source_citation} — {_fmt(f.a)}; {b.source_citation} — {_fmt(f.b)}"


def analyze_conflicts(relationships: list[Relationship], claims: dict[str, Claim]) -> list[ConflictAnalysis]:
    out: list[ConflictAnalysis] = []
    for rel in relationships:
        if not _is_conflict(rel):
            continue
        a, b = claims[rel.target_claim_id], claims[rel.source_claim_id]
        differing = sorted([f for f in rel.factors if f.status == "different"], key=lambda f: _PRIORITY.index(f.factor))
        shared = [f for f in rel.factors if f.status in ("same", "overlap")]
        unknown = [f.factor for f in rel.factors if f.status == "unknown"]

        if rel.relation == RelationType.QUALIFIES and differing:
            verdict = ConflictVerdict.CONTEXT_DEPENDENT
            names = ", ".join(f.factor for f in differing[:3])
            headline = f"Apparent conflict likely explained by different {names}"
            lines = [_factor_sentence(f, a, b) for f in differing[:4]]
            explanation = (
                f"These findings may not actually be contradictory: the studies differ in {names}. "
                + " ".join(f"({i + 1}) {line}." for i, line in enumerate(lines))
                + " Each claim may hold within its own setting; a study holding these factors fixed would be needed to test a direct conflict."
            )
            confidence = rel.confidence
        elif rel.uncertain or not [f for f in shared + differing if f.factor not in ("model / method", "baseline")]:
            verdict = ConflictVerdict.INSUFFICIENT_EVIDENCE
            if rel.uncertain and differing:
                # the conditions are known; it is the link itself that is weak
                names = ", ".join(f.factor for f in differing[:3])
                headline = "Possible disagreement — low-confidence link"
                explanation = (
                    "The claims may point in opposite directions, but the relationship between them is uncertain, so "
                    f"ClaimGraph does not treat this as an established contradiction. If they do conflict, the studies "
                    f"also differ in {names}, which could explain it."
                )
            else:
                headline = "Possible disagreement — context too sparse to judge"
                explanation = (
                    "The claims appear to point in opposite directions, but the source passages do not state "
                    f"{', '.join(unknown[:5]) or 'enough experimental detail'}. ClaimGraph therefore does not treat this "
                    "as an established contradiction."
                )
            if any(c.claim_type.value == "background" for c in (a, b)):
                explanation += " At least one side is a belief attributed to prior work, not a result reported by that paper."
            confidence = min(rel.confidence, 0.45)
        else:
            verdict = ConflictVerdict.GENUINE
            names = ", ".join(f.factor for f in shared[:4]) or "the stated conditions"
            headline = "Opposing findings under comparable conditions"
            explanation = (
                f"These studies evaluate substantially similar conditions ({names}) but report opposing findings. "
                + (f"Remaining differences: {', '.join(f.factor for f in differing)}. " if differing else "")
                + (f"Not stated in either passage: {', '.join(unknown[:4])}, which could still explain the gap." if unknown else "")
            )
            confidence = rel.confidence

        out.append(ConflictAnalysis(
            conflict_id="cf_" + hashlib.sha1(rel.relation_id.encode()).hexdigest()[:10],
            relation_id=rel.relation_id,
            claim_a_id=a.claim_id,
            claim_b_id=b.claim_id,
            verdict=verdict,
            headline=headline,
            explanation=explanation.strip(),
            differing_factors=differing,
            shared_factors=shared,
            unknown_factors=unknown,
            evidence_passage_ids=[p.passage_id for p in a.evidence + b.evidence],
            confidence=round(confidence, 3),
        ))
    order = {ConflictVerdict.GENUINE: 0, ConflictVerdict.CONTEXT_DEPENDENT: 1, ConflictVerdict.INSUFFICIENT_EVIDENCE: 2}
    out.sort(key=lambda c: (order[c.verdict], -c.confidence))
    return out
