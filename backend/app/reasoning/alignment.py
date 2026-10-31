"""Grounded LLM pair analysis: stance + evidence-constrained context comparison.

Surface rules can only compare context values they have a lexicon for. For an open
vocabulary of datasets, regimes and populations ClaimGraph instead asks an LLM to *read*
the two passages and fill in a structured comparison, and then keeps the decision itself
in the deterministic rules of :mod:`app.reasoning.relations`:

1. **Stance.** Do the two claims, taken at face value, reach conflicting or matching
   conclusions about the same relationship? The model is told to ignore differences in
   setting here — whether a disagreement is *real* is exactly what the context gate decides.
2. **Context table.** For each context factor, the value each claim was established under,
   with a verbatim quote from that paper's text.

The LLM's output is never trusted as evidence. Code enforces the grounding rules:

* a value whose quote cannot be found in that paper's text is dropped;
* a factor grounded on one side only is ``unknown`` — never ``different`` — so missing
  information cannot manufacture an explanation;
* a value grounded only in the abstract (not the claim's own passage) is marked inherited
  and counts half, as for the rule-based context;
* a stated difference the model judges incidental to the findings is kept, but as
  ``overlap`` rather than ``different``, so it cannot by itself turn a conflict into QUALIFIES.

Any provider failure, malformed output or exhausted budget yields ``None`` for that pair
and the rule-based view stands.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.providers.llm import BudgetedLLM
from app.reasoning.relations import FACTORS, PairAnalyzer, PairCandidate, PairView
from app.schemas.domain import Claim, FactorComparison, Paper

log = logging.getLogger(__name__)

FACTOR_GUIDE: list[tuple[str, str]] = [
    ("dataset", "the benchmark, dataset, corpus or cohort the result was obtained on"),
    ("data scale", "how much (pre-)training data or how large a dataset was used"),
    ("experimental conditions", "the regime: e.g. pre-trained vs from scratch, zero-shot vs fine-tuned, training "
                                "duration or budget, hyperparameter-tuning protocol, dose, follow-up length"),
    ("training & design", "specific techniques used: augmentation, regularisation, optimiser, architecture changes"),
    ("task", "the task or problem type"),
    ("population", "who or what was studied: participants, model sizes or classes, domains"),
    ("evaluation metric", "how the outcome was measured"),
    ("model / method", "the model, method or intervention the claim is about"),
    ("baseline", "what it is compared against"),
    ("sample size", "number of subjects, runs or examples"),
    ("methodology", "study type: e.g. randomised trial, ablation, theoretical analysis, meta-analysis"),
    ("time period", "when the data were collected"),
]
assert [name for name, _ in FACTOR_GUIDE] and {name for name, _ in FACTOR_GUIDE} == {label for label, _, _ in FACTORS}

STANCES = ["oppose", "agree", "partial", "unrelated"]

# worked examples for the stance step, deliberately from domains unrelated to any evaluation topic
STANCE_EXAMPLES: list[tuple[str, str, str]] = [
    ("Dropout improves the accuracy of CNNs on small image datasets.",
     "With billions of training images, dropout no longer improves accuracy.",
     "subject dropout / dropout, outcome accuracy / accuracy: same relationship. 'dropout improves accuracy' versus "
     "'dropout does not improve accuracy': oppose (the data scale differs, which step 3 records)."),
    ("Fine-tuned BERT outperforms LSTMs on English sentiment classification.",
     "BERT also outperforms LSTMs on low-resource languages.",
     "subject BERT versus LSTMs / BERT versus LSTMs, outcome accuracy / accuracy: same relationship. Same conclusion in "
     "a different setting: agree."),
    ("BERT outperforms LSTMs on sentiment classification.",
     "Given enough training data, LSTMs match BERT.",
     "same relationship (BERT versus LSTMs, accuracy). An advantage versus no advantage: oppose."),
    ("Caffeine improves reaction time.",
     "Caffeine improves reaction time only in sleep-deprived participants.",
     "same relationship (caffeine, reaction time). The second conclusion is narrower: partial."),
    ("Our pruning method retains 99% of the accuracy at 90% sparsity.",
     "Our quantisation method retains 99% of the accuracy at 8 bits.",
     "subject pruning method / quantisation method: different subjects, not the same relationship: unrelated."),
    ("Method X halves the training time.",
     "Method X improves accuracy by two points.",
     "outcome training time / accuracy: different outcomes, not the same relationship: unrelated."),
]

SYSTEM = (
    "You compare two claims taken from two different scientific papers. You do not decide which paper is right. "
    "Use only the text you are given; never use outside knowledge about these papers.\n"
    "Judge the two CLAIM sentences themselves, not the papers. The abstracts and passages are there only so that you "
    "can tell what a claim refers to (for example what 'our method' is) and under which conditions it was obtained.\n\n"
    "STEP 1 - same relationship? For each claim name its \"subject\" (the method, model family, intervention or "
    "factor it is about) and its \"outcome\" (what is measured or concluded about it), in a few words, WITHOUT any "
    "dataset, setting or condition. The claims are about the same relationship when they have the same subject "
    "(compared against the same kind of alternative, if any) and the same kind of outcome. Conditions are not part "
    "of the relationship: 'drug D lowers blood pressure in adults' and 'drug D does not lower blood pressure in "
    "children' are the same relationship (D -> blood pressure) under different conditions. But sharing a topic, a "
    "research area or a dataset is not enough: 'our method A beats baseline Z' and 'our method B beats baseline Z' "
    "have different subjects, and a claim about accuracy and a claim about training cost have different outcomes. "
    "If the relationship is not the same, set \"same_relationship\" to false and \"stance\" to \"unrelated\".\n\n"
    "STEP 2 - stance. If the relationship is the same, restate what each claim concludes about it in one short "
    "clause (\"conclusion_a\", \"conclusion_b\"), then compare the two conclusions at face value, IGNORING any "
    "difference in datasets, settings or conditions (step 3 records those):\n"
    '- "agree": the same conclusion (the same direction of effect).\n'
    '- "oppose": conflicting conclusions (opposite direction of effect, or one reports an effect or advantage that '
    "the other denies or finds absent).\n"
    '- "partial": the conclusions overlap, but one is explicitly narrower, weaker or more conditional than the other '
    "(for example 'X always helps' versus 'X helps only for large models').\n"
    "Two claims that reach the same conclusion under different conditions AGREE - they do not oppose, and they are "
    "not \"partial\" merely because the settings differ.\n\n"
    "Examples for steps 1 and 2:\n"
    + "\n".join(f"- A: '{a}' B: '{b}' -> {verdict}" for a, b, verdict in STANCE_EXAMPLES)
    + "\n\n"
    "STEP 3 - context. For each context factor, report the value under which EACH claim was established, but only "
    "if that paper's text states it. For every value give a short quote copied EXACTLY (at most 15 words) from that "
    "paper's text that contains it. If a paper's text does not state a factor, use null for both its value and its "
    "quote. Never infer a value from the other paper or from background knowledge. List only factors that at least "
    "one paper states, most relevant first, at most eight. For an unrelated pair the list may be empty.\nFactors:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in FACTOR_GUIDE)
    + "\n\nFor each listed factor set \"status\" to \"same\", \"different\" or \"overlap\" (partly the same) when both "
    "papers state it, and to \"one_sided\" otherwise. Set \"material\" to true only if the difference could plausibly "
    "explain why the two findings differ (or, when they agree, shows that the finding holds in a genuinely different "
    "setting); false if it is incidental."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "subject_a": {"type": "string"},
        "outcome_a": {"type": "string"},
        "subject_b": {"type": "string"},
        "outcome_b": {"type": "string"},
        "same_relationship": {"type": "boolean"},
        "conclusion_a": {"type": ["string", "null"]},
        "conclusion_b": {"type": ["string", "null"]},
        "stance": {"type": "string", "enum": STANCES},
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "factor": {"type": "string", "enum": [name for name, _ in FACTOR_GUIDE]},
                    "a_value": {"type": ["string", "null"]},
                    "a_quote": {"type": ["string", "null"]},
                    "b_value": {"type": ["string", "null"]},
                    "b_quote": {"type": ["string", "null"]},
                    "status": {"type": "string", "enum": ["same", "different", "overlap", "one_sided"]},
                    "material": {"type": "boolean"},
                },
                "required": ["factor", "a_value", "a_quote", "b_value", "b_quote", "status", "material"],
            },
        },
    },
    "required": ["subject_a", "outcome_a", "subject_b", "outcome_b", "same_relationship", "conclusion_a", "conclusion_b",
                 "stance", "factors"],
}

MAX_TOKENS = 1500  # a full twelve-factor table with quotes is ~600-900 tokens
QUOTE_MATCH = 90  # rapidfuzz partial_ratio a quote must reach against the source text
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower().replace("‑", "-").replace("–", "-").replace("’", "'")).strip()


def quote_grounded(quote: str | None, source: str) -> bool:
    """True when ``quote`` (near-)verbatim occurs in ``source``."""
    q = _norm(quote or "")
    if len(q) < 3:
        return False
    src = _norm(source)
    return q in src or fuzz.partial_ratio(q, src) >= QUOTE_MATCH


@dataclass
class GroundingStats:
    """How much of the model's context table survived the grounding checks (reported in experiments)."""

    values: int = 0
    grounded_local: int = 0
    grounded_paper: int = 0
    dropped: int = 0
    downgraded_immaterial: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class PairTexts:
    """The text of one side that the model sees, and that quotes are checked against."""

    local: str  # the claim and its evidence passage
    paper: str  # title + abstract


def claim_texts(claim: Claim, paper: Paper | None) -> PairTexts:
    local = " ".join([claim.claim_text, *(p.text for p in claim.evidence)])
    paper_level = f"{paper.title}. {paper.abstract or ''}" if paper else ""
    return PairTexts(local=local, paper=paper_level)


def render_pair(a: Claim, b: Claim, papers: dict[str, Paper], abstract_chars: int = 1500, question: str = "") -> str:
    def side(tag: str, c: Claim) -> str:
        p = papers.get(c.paper_id)
        title = p.title if p else "unknown"
        abstract = (p.abstract or "")[:abstract_chars] if p else ""
        passage = " ".join(e.text for e in c.evidence)
        return (f"PAPER {tag}: {c.source_citation} - {title}\n"
                f"Abstract {tag}: {abstract or 'not available'}\n"
                f"Passage {tag} (section: {c.section}): {passage}\n"
                f"CLAIM {tag}: {c.claim_text}")

    head = f"Topic of the literature review: {question}\n\n" if question else ""
    return head + side("A", a) + "\n\n" + side("B", b)


def _stance_note(out: dict) -> str | None:
    a, b = str(out.get("conclusion_a") or "").strip(), str(out.get("conclusion_b") or "").strip()
    if not (a and b):
        return None
    return f"A concludes: {a}. B concludes: {b}."[:500]


def view_from_output(out: dict, a: Claim, b: Claim, papers: dict[str, Paper],
                     stats: GroundingStats | None = None, enforce_grounding: bool = True) -> PairView | None:
    """Validate a model response and turn it into a :class:`PairView` (``None`` if unusable)."""
    stance = str(out.get("stance", "")).lower()
    if stance not in STANCES:
        return None
    if out.get("same_relationship") is False:
        stance = "unrelated"  # the model's own first answer wins over an inconsistent stance label
    stats = stats or GroundingStats()
    ta, tb = claim_texts(a, papers.get(a.paper_id)), claim_texts(b, papers.get(b.paper_id))
    by_factor: dict[str, FactorComparison] = {}
    for item in out.get("factors") or []:
        if not isinstance(item, dict) or item.get("factor") not in {n for n, _ in FACTOR_GUIDE} or item["factor"] in by_factor:
            continue
        sides = []
        inherited = False
        for value, quote, texts, claim in ((item.get("a_value"), item.get("a_quote"), ta, a),
                                           (item.get("b_value"), item.get("b_quote"), tb, b)):
            value = str(value).strip() if value not in (None, "", "null") else ""
            if not value:
                sides.append(("", None))
                continue
            stats.values += 1
            if not enforce_grounding or quote_grounded(quote, texts.local):
                stats.grounded_local += 1
                sides.append((value, quote))
            elif quote_grounded(quote, texts.paper):
                stats.grounded_paper += 1
                inherited = inherited or claim.section != "abstract"
                sides.append((value, quote))
            else:
                stats.dropped += 1
                sides.append(("", None))
        (va, qa), (vb, qb) = sides
        note = None
        if not va or not vb:
            status = "unknown"  # stated by one side only: unknown, never different
        else:
            status = item.get("status") if item.get("status") in ("same", "different", "overlap") else "unknown"
            if status == "different" and item.get("material") is False:
                status = "overlap"
                note = "stated difference judged incidental to the findings"
                stats.downgraded_immaterial += 1
        by_factor[item["factor"]] = FactorComparison(
            factor=item["factor"], a=[va] if va else [], b=[vb] if vb else [], status=status,  # type: ignore[arg-type]
            inherited=inherited and status != "unknown", a_quote=qa, b_quote=qb, note=note)
    factors = [by_factor.get(label) or FactorComparison(factor=label, status="unknown") for label, _, _ in FACTORS]
    related = stance != "unrelated"
    return PairView(
        stance={"oppose": "oppose", "agree": "agree", "partial": "parity_mismatch"}.get(stance),
        comparable=related,
        same_direction=stance == "agree",
        same_subject=related,
        shared_object=related,
        factors=factors,
        confident=True,
        source="llm",
        note=_stance_note(out),
    )


@dataclass
class LLMPairAnalyzer(PairAnalyzer):
    papers: dict[str, Paper]
    concurrency: int = 4
    question: str = ""  # optional review topic to show the model; off by default (it biased dev runs towards 'related')
    max_tokens: int = MAX_TOKENS
    enforce_grounding: bool = True  # False only for the "ungrounded" ablation
    stats: GroundingStats = field(default_factory=GroundingStats)

    def analyze_one(self, pair: PairCandidate, llm: BudgetedLLM) -> PairView | None:
        out = llm.complete_json(SYSTEM, render_pair(pair.a, pair.b, self.papers, question=self.question), SCHEMA,
                                max_tokens=self.max_tokens)
        if not out:
            self.stats.failures += 1
            return None
        try:
            view = view_from_output(out, pair.a, pair.b, self.papers, self.stats, self.enforce_grounding)
        except Exception:  # noqa: BLE001 — a malformed response must never break an analysis
            log.exception("unusable pair-analysis response")
            view = None
        if view is None:
            self.stats.failures += 1
        return view

    def analyze(self, pairs: list[PairCandidate], llm: BudgetedLLM) -> list[PairView | None]:
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as pool:
            return list(pool.map(lambda p: self.analyze_one(p, llm), pairs))
