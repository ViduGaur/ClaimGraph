"""Conditional-prediction frames: one structured record per claim, compared in code (experiment E3).

A frame is read from ONE claim, so what it says cannot be bent toward the other claim of a pair:

    intervention X  (vs comparator Z)  ->  effect on outcome Y   | under conditions C

The LLM extracts the record and quotes the text. Code checks the quotes and decides everything after that: whether two
frames concern the same intervention and outcome, whether their effects are incompatible, whether their stated
conditions are comparable, and so the relation and conflict status.

Not wired into the pipeline; evaluated in ``app.evaluation.layer_study``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.reasoning.alignment import FACTOR_GUIDE, quote_grounded
from app.reasoning.comparability import NAMED_FACTORS, NUMERIC_FACTORS, deterministic_status
from app.schemas.domain import Claim, Paper

EFFECTS = ["positive", "negative", "none", "mixed", "not_stated"]
# "model / method" and "baseline" are the intervention and the comparator of a frame; the other ten are conditions
CONDITION_FACTORS = [name for name, _ in FACTOR_GUIDE if name not in ("model / method", "baseline")]

# worked examples, deliberately from domains unrelated to any evaluation topic
EXAMPLES = [
    ("In adults over 60, the new drug D-12 lowered systolic blood pressure by 9 mmHg more than placebo.",
     '{"intervention": "drug D-12", "intervention_is_new": true, "comparator": "placebo", "outcome": "systolic blood '
     'pressure reduction", "effect": "positive", "effect_quote": "lowered systolic blood pressure by 9 mmHg more than '
     'placebo", "conditional": true, "conditions": [{"factor": "population", "value": "adults over 60", "quote": "In '
     'adults over 60"}]}'),
    ("We found no difference in wheat yield between no-till and conventional tillage.",
     '{"intervention": "no-till farming", "intervention_is_new": false, "comparator": "conventional tillage", '
     '"outcome": "wheat yield", "effect": "none", "effect_quote": "no difference in wheat yield", "conditional": false, '
     '"conditions": []}'),
    ("We describe a protocol for measuring soil carbon with a handheld spectrometer.",
     '{"intervention": "handheld spectrometer protocol", "intervention_is_new": true, "comparator": null, "outcome": '
     '"soil carbon measurement", "effect": "not_stated", "effect_quote": null, "conditional": false, "conditions": []}'),
]

SYSTEM = (
    "You read ONE claim sentence taken from a scientific paper and describe what it asserts as a structured record. "
    "You do not judge whether the claim is right, and you do not compare it with any other paper. Use only the text "
    "you are given; never use outside knowledge.\n"
    "Describe the CLAIM sentence itself. The passage and the abstract are there only so that you can resolve what a "
    "phrase in the claim refers to ('our method', 'this', 'the latter') and under which conditions the claim was "
    "obtained. Do not replace the claim with the overall conclusion of the paper.\n\n"
    "Fields:\n"
    '- "intervention": the method, model family, technique, treatment or factor whose effect the claim is about, as '
    "a short noun phrase with references resolved (the method's name instead of 'our method'). No datasets, "
    "settings or conditions.\n"
    '- "intervention_is_new": true if the intervention is a method, variant or modification that THIS paper '
    "proposes or introduces; false if it is an existing method or a general factor.\n"
    '- "comparator": what the claim compares the intervention against (a baseline, an alternative, or the absence '
    "of the intervention), or null if the claim makes no comparison.\n"
    '- "outcome": the quantity or property the claim is about, as a short noun phrase without direction words. '
    "Where a standard desirable form exists use it ('accuracy' rather than 'error').\n"
    '- "effect": what the claim says the intervention does to the outcome (relative to the comparator, if any):\n'
    '  "positive": it improves or increases the outcome, or is better than the comparator;\n'
    '  "negative": it worsens or decreases the outcome, or is worse than the comparator;\n'
    '  "none": no effect, no (significant) difference, or a negligible one;\n'
    '  "mixed": the claim itself says that the effect depends on or varies with conditions, or reports both a '
    "benefit and a harm;\n"
    '  "not_stated": the claim does not assert an effect of an intervention on an outcome (it describes a method, '
    "explains a mechanism, states a goal, or reports what was analysed).\n"
    '- "effect_quote": the words of the CLAIM sentence, copied EXACTLY (at most 20 words), that state this effect; '
    'null when the effect is "not_stated".\n'
    '- "conditional": true if the claim sentence itself restricts the effect to particular conditions (\'for small '
    "models', 'in the low-data regime', 'when ...'); otherwise false.\n"
    '- "conditions": the conditions under which the claim was established, one entry per context factor that this '
    "paper's text states. For each give the value and a short quote copied EXACTLY (at most 15 words) from the "
    "claim, the passage or the abstract that contains it. Never infer a value. Most relevant first, at most eight; "
    "the list may be empty.\nCondition factors:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in FACTOR_GUIDE if name in CONDITION_FACTORS)
    + "\n\nExamples:\n"
    + "\n".join(f"CLAIM: {claim}\n{record}" for claim, record in EXAMPLES)
    + "\n\nAnswer with one JSON object with exactly these fields."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "intervention": {"type": "string"},
        "intervention_is_new": {"type": "boolean"},
        "comparator": {"type": ["string", "null"]},
        "outcome": {"type": "string"},
        "effect": {"type": "string", "enum": EFFECTS},
        "effect_quote": {"type": ["string", "null"]},
        "conditional": {"type": "boolean"},
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"factor": {"type": "string", "enum": CONDITION_FACTORS}, "value": {"type": "string"},
                               "quote": {"type": "string"}},
                "required": ["factor", "value", "quote"],
            },
        },
    },
    "required": ["intervention", "intervention_is_new", "comparator", "outcome", "effect", "effect_quote", "conditional",
                 "conditions"],
}
MAX_TOKENS = 700


def render_claim(claim: Claim, paper: Paper | None, abstract_chars: int = 1500) -> str:
    title = paper.title if paper else "unknown"
    abstract = (paper.abstract or "")[:abstract_chars] if paper else ""
    passage = " ".join(e.text for e in claim.evidence)
    return (f"PAPER: {claim.source_citation} - {title}\n"
            f"Abstract: {abstract or 'not available'}\n"
            f"Passage (section: {claim.section}): {passage}\n"
            f"CLAIM: {claim.claim_text}")


@dataclass(frozen=True)
class Condition:
    factor: str
    value: str
    quote: str
    level: str  # "claim": quoted from the claim sentence; "passage": from its passage; "paper": from the abstract


@dataclass(frozen=True)
class ClaimFrame:
    claim_id: str
    intervention: str
    is_new: bool
    comparator: str
    outcome: str
    effect: str  # one of EFFECTS, as extracted
    effect_grounded: bool  # the effect quote is found in the claim sentence
    conditional: bool
    conditions: dict[str, Condition] = field(default_factory=dict)

    @property
    def states_effect(self) -> bool:
        """An effect that the claim sentence itself can be shown to state."""
        return self.effect != "not_stated" and self.effect_grounded


def _text(v) -> str:
    return "" if v in (None, "null") else str(v).strip()


def frame_from_output(out: dict | None, claim: Claim, paper: Paper | None) -> ClaimFrame | None:
    """Validate a model response and ground its quotes (``None`` if unusable)."""
    if not isinstance(out, dict) or str(out.get("effect", "")).lower() not in EFFECTS or not _text(out.get("intervention")):
        return None
    effect = str(out["effect"]).lower()
    passage = " ".join(e.text for e in claim.evidence)
    paper_text = f"{paper.title}. {paper.abstract or ''}" if paper else ""
    conditions: dict[str, Condition] = {}
    for item in out.get("conditions") or []:
        if not isinstance(item, dict) or item.get("factor") not in CONDITION_FACTORS or item["factor"] in conditions:
            continue
        value, quote = _text(item.get("value")), _text(item.get("quote"))
        if not value:
            continue
        if quote_grounded(quote, claim.claim_text):
            level = "claim"
        elif quote_grounded(quote, passage):
            level = "passage"
        elif quote_grounded(quote, paper_text):
            level = "paper"
        else:
            continue  # a value whose quote is not in the paper's text is dropped
        conditions[item["factor"]] = Condition(item["factor"], value, quote, level)
    return ClaimFrame(
        claim_id=claim.claim_id, intervention=_text(out.get("intervention")), is_new=out.get("intervention_is_new") is True,
        comparator=_text(out.get("comparator")), outcome=_text(out.get("outcome")), effect=effect,
        effect_grounded=effect != "not_stated" and quote_grounded(_text(out.get("effect_quote")), claim.claim_text),
        conditional=out.get("conditional") is True, conditions=conditions)


# ------------------------------------------------------------------------------------------- comparison
_GENERIC = {"the", "a", "an", "of", "and", "or", "for", "in", "on", "with", "to", "based", "method", "methods", "model",
            "models", "approach", "approaches", "algorithm", "algorithms", "technique", "techniques", "training", "our",
            "proposed", "new", "novel", "using", "via", "s"}
_FLIP = {"positive": "negative", "negative": "positive"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _GENERIC}


def _name_variants(text: str) -> list[set[str]]:
    """Token sets of a method name and of each parenthesised alias: 'sharpness-aware minimization (SAM)' has two."""
    inside = re.findall(r"\(([^)]*)\)", text)
    outside = re.sub(r"\([^)]*\)", " ", text)
    return [t for t in (_tokens(outside), *(_tokens(x) for x in inside)) if t]


def names_match(new_name: str, other: str) -> bool:
    """Does ``other`` name the newly proposed method ``new_name``? True when every token of the name, or of one of
    its aliases, occurs in the other phrase. 'ND-Adam' is not named by 'Adam-like adaptive algorithms'."""
    other_tokens = set().union(*_name_variants(other)) if _name_variants(other) else set()
    return any(v <= other_tokens for v in _name_variants(new_name))


def effect_relation(a: str, b: str) -> str:
    """'agree' | 'opposite' (benefit vs harm) | 'effect_vs_none' | 'mixed' (either claim says: it depends)."""
    if "mixed" in (a, b):
        return "mixed"
    if a == b:
        return "agree"
    return "effect_vs_none" if "none" in (a, b) else "opposite"


def condition_status(factor: str, a: str, b: str) -> str:
    """'same' | 'overlap' | 'different' for two stated values of one condition factor.

    Names and quantities are compared as in ``comparability.deterministic_status``; free-text factors by their
    content words (identical or contained: same; some shared: overlap; none shared: different)."""
    if factor in NAMED_FACTORS or factor in NUMERIC_FACTORS:
        det = deterministic_status(factor, [a], [b])
        if det:
            return det
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return "different"
    if ta <= tb or tb <= ta:
        return "same"
    return "overlap" if ta & tb else "different"


@dataclass(frozen=True)
class FrameComparison:
    both_effect: bool  # both claim sentences state an effect
    s_intervention: float  # similarity of the aligned interventions
    s_outcome: float
    alignment: str  # "direct" | "a_comparator" | "b_comparator": which phrase of each claim was aligned
    new_mismatch: bool  # one claim is about a newly proposed method that the other does not name
    effects: str | None  # effect_relation after alignment; None without two effects
    conditional: bool  # at least one claim sentence restricts itself to particular conditions
    shared: dict[str, str]  # condition factor stated by both claims -> same | overlap | different
    one_sided: list[str]  # condition factors stated by one claim only

    @property
    def incompatible(self) -> bool:
        return self.effects in ("opposite", "effect_vs_none")


def compare_frames(a: ClaimFrame, b: ClaimFrame, sim: Callable[[str, str], float], *, allow_swap: bool = True) -> FrameComparison:
    """Deterministic comparison of two frames. ``sim`` is a phrase similarity in [0, 1] (embedding cosine).

    ``allow_swap=False`` always aligns intervention with intervention (a sensitivity check: phrase similarity
    cannot reliably tell a method from its own comparator, e.g. large-batch from small-batch training)."""
    # which phrases to align: 'Adam is worse than SGD' and 'SGD beats Adam' are about the same contrast with the
    # roles swapped, so the comparator of one claim may be the intervention of the other (and its effect flips)
    options = [("direct", sim(a.intervention, b.intervention), a.effect, b.effect)]
    if a.comparator and allow_swap:
        options.append(("a_comparator", sim(a.comparator, b.intervention), _FLIP.get(a.effect, a.effect), b.effect))
    if b.comparator and allow_swap:
        options.append(("b_comparator", sim(a.intervention, b.comparator), a.effect, _FLIP.get(b.effect, b.effect)))
    alignment, s_int, ea, eb = max(options, key=lambda o: (o[1], o[0] == "direct"))
    both = a.states_effect and b.states_effect
    subj_a = a.comparator if alignment == "a_comparator" else a.intervention
    subj_b = b.comparator if alignment == "b_comparator" else b.intervention
    new_mismatch = ((a.is_new and alignment != "a_comparator" and not names_match(a.intervention, subj_b))
                    or (b.is_new and alignment != "b_comparator" and not names_match(b.intervention, subj_a)))
    shared = {f: condition_status(f, a.conditions[f].value, b.conditions[f].value)
              for f in CONDITION_FACTORS if f in a.conditions and f in b.conditions}
    one_sided = [f for f in CONDITION_FACTORS if (f in a.conditions) != (f in b.conditions)]
    return FrameComparison(both, float(s_int), float(sim(a.outcome, b.outcome)), alignment, new_mismatch,
                           effect_relation(ea, eb) if both else None, a.conditional or b.conditional, shared, one_sided)


def frame_relation(c: FrameComparison, threshold: float) -> str:
    """Relation of two claims from their frames: UNRELATED | SUPPORTS | CONTRADICTS | QUALIFIES.

    Same relationship: the aligned interventions and the outcomes are at least ``threshold`` similar and no newly
    proposed method is involved that the other claim does not name. Then the effects decide: the same effect
    SUPPORTS, incompatible effects CONTRADICT (at face value), and a claim that says "it depends" QUALIFIES.
    """
    if not c.both_effect or c.new_mismatch or c.s_intervention < threshold or c.s_outcome < threshold:
        return "UNRELATED"
    return {"agree": "SUPPORTS", "opposite": "CONTRADICTS", "effect_vs_none": "CONTRADICTS", "mixed": "QUALIFIES"}[c.effects or "agree"]


def frame_status(c: FrameComparison, threshold: float) -> str:
    """Conflict status from two frames: none | genuine | context_dependent | insufficient.

    Only a face-value contradiction has a status. A condition that both claims state and that differs, or a claim
    that restricts itself to particular conditions, makes the conflict context-dependent. With no condition
    stated by both claims there is nothing to compare: insufficient. Otherwise the stated conditions match:
    genuine.
    """
    if frame_relation(c, threshold) != "CONTRADICTS":
        return "none"
    if any(s == "different" for s in c.shared.values()) or c.conditional:
        return "context_dependent"
    return "genuine" if c.shared else "insufficient"
