"""Cross-paper claim comparison and relationship classification.

For every candidate pair of claims from *different* papers ClaimGraph combines
four independent signals, then applies explicit, inspectable decision rules:

1. semantic similarity of normalised claims (bi-encoder embeddings)
2. natural-language inference in both directions (DeBERTa-v3 NLI cross-encoder)
3. stance/polarity: does one claim say X beats Y while the other says the reverse?
4. context comparison: dataset, data scale, conditions, training, metric, task, …

The rules are designed so that a contradiction is only asserted when the
contexts are known *and* comparable; opposing claims under different
conditions become QUALIFIES (+ a context-dependent conflict record), and
opposing claims with unknown context are flagged as uncertain.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from app.extraction.claims import family
from app.providers.llm import BudgetedLLM
from app.providers.models import ModelBundle
from app.schemas.domain import (
    Claim,
    ClaimContext,
    ClaimType,
    FactorComparison,
    Paper,
    Polarity,
    RelationSignals,
    RelationType,
    Relationship,
)

log = logging.getLogger(__name__)

# (label shown to users, ClaimContext attribute, weight in divergence score)
FACTORS: list[tuple[str, str, float]] = [
    ("data scale", "data_scale", 1.5),
    ("dataset", "datasets", 1.2),
    ("experimental conditions", "conditions", 1.3),
    ("training & design", "training", 1.0),
    ("task", "task", 1.0),
    ("population", "population", 1.2),
    ("evaluation metric", "metrics", 0.8),
    ("model / method", "models", 0.6),
    ("baseline", "baselines", 0.6),
    ("sample size", "sample_size", 0.4),
    ("methodology", "methodology", 0.5),
    ("time period", "time_period", 0.5),
]
_FAMILY_FACTORS = {"models", "baselines"}
_SUBJECT_FACTORS = {"model / method", "baseline"}
_EXCLUSIVE_CONDITIONS = [
    ({"large-scale pre-training", "pre-training"}, {"trained from scratch", "no external data"}),
    ({"small datasets"}, {"large-scale pre-training"}),
]


def _as_list(v) -> list[str]:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [str(v)]


def compare_contexts(a: ClaimContext, b: ClaimContext, a_inherited: list[str] | None = None,
                     b_inherited: list[str] | None = None) -> list[FactorComparison]:
    inherited = set(a_inherited or []) | set(b_inherited or [])
    out: list[FactorComparison] = []
    for label, attr, _ in FACTORS:
        va, vb = _as_list(getattr(a, attr)), _as_list(getattr(b, attr))
        if not va or not vb:
            status = "unknown"
        else:
            sa = {family(x) for x in va} if attr in _FAMILY_FACTORS else set(va)
            sb = {family(x) for x in vb} if attr in _FAMILY_FACTORS else set(vb)
            if sa == sb:
                status = "same"
            elif sa & sb:
                status = "overlap"
                if attr == "conditions" and any((sa & x and sb & y) or (sa & y and sb & x) for x, y in _EXCLUSIVE_CONDITIONS):
                    status = "different"  # e.g. one pre-trains at scale, the other trains from scratch
            else:
                status = "different"
        out.append(FactorComparison(factor=label, a=va, b=vb, status=status,  # type: ignore[arg-type]
                                    inherited=attr in inherited and status != "unknown"))
    return out


def context_divergence(factors: list[FactorComparison]) -> float | None:
    """0 = same known conditions … 1 = entirely different; None = nothing comparable is known.

    Factors whose values were inherited from paper-level text (rather than the
    claim's own passage) count half, since they may not apply to this claim.
    """
    weights = {label: w for label, _, w in FACTORS}
    num = den = 0.0
    for f in factors:
        if f.status == "unknown":
            continue
        w = weights[f.factor] * (0.5 if f.inherited else 1.0)
        num += w * {"same": 0.0, "overlap": 0.4, "different": 1.0}[f.status]
        den += w
    return None if den == 0 else num / den


def stance(claim: Claim) -> tuple[tuple[str, str | None], int] | None:
    """(subject family, object family) → +1 subject better / -1 worse / 0 parity."""
    if not claim.subject or claim.polarity == Polarity.NONE:
        return None
    subj = family(claim.subject)
    obj = family(claim.context.baselines[0]) if claim.context.baselines else None
    if obj == subj:
        obj = None
    sign = {Polarity.POSITIVE: 1, Polarity.NEGATIVE: -1, Polarity.NEUTRAL: 0}[claim.polarity]
    if obj and obj < subj:  # canonical ordering so (a,b,+1) ≡ (b,a,-1)
        return (obj, subj), -sign
    return (subj, obj), sign


def stance_relation(a: Claim, b: Claim) -> str | None:
    """'oppose' | 'agree' | 'parity_mismatch' | None when stances are not comparable."""
    sa, sb = stance(a), stance(b)
    if not sa or not sb:
        return None
    (ka, xa), (kb, xb) = sa, sb
    if ka[0] != kb[0]:
        return None
    if ka[1] != kb[1] and ka[1] is not None and kb[1] is not None:
        return None
    if xa * xb < 0:
        return "oppose"
    if xa == xb and xa != 0:
        return "agree"
    if (xa == 0) != (xb == 0) and ka[1] is not None and kb[1] is not None:
        return "parity_mismatch"
    return None


def polarity_conflict(a: Claim, b: Claim) -> tuple[bool, bool]:
    """Returns (opposing, parity_vs_directional)."""
    rel = stance_relation(a, b)
    return rel == "oppose", rel == "parity_mismatch"


@dataclass(frozen=True)
class RuleConfig:
    """Which components of the decision procedure are active. Defaults = the full system.

    The switches exist for ablation studies (see ``app.evaluation``):

    * ``use_stance`` — stance reasoning: opposing/agreeing direction of effect on a shared comparison,
      and the guards that stop NLI from calling "X improves Y" vs "Z improves Y" a contradiction.
    * ``use_context_gate`` — condition a contradiction on comparable context. When off, every
      disagreement signal becomes CONTRADICTS (no QUALIFIES downgrade, no abstention).
    """

    use_stance: bool = True
    use_context_gate: bool = True


@dataclass
class PairView:
    """Everything the decision rules need to know about a pair besides similarity and NLI.

    Produced either by surface rules (:func:`rule_view`) or by the grounded LLM pair analyser
    (:mod:`app.reasoning.alignment`); the decision rules are the same for both.
    """

    stance: str | None = None  # 'oppose' | 'agree' | 'parity_mismatch' | None (not comparable / unknown)
    comparable: bool | None = None  # False = the claims answer different questions; None = not assessed
    same_direction: bool = False  # both report the same direction of effect (possibly for different things)
    same_subject: bool = False
    shared_object: bool = False
    factors: list[FactorComparison] = field(default_factory=list)
    confident: bool = False  # stance is a semantic judgement, so it needs no NLI corroboration
    source: str = "rules"  # 'rules' | 'llm'
    note: str | None = None  # the analyser's one-sentence rationale, if any


def rule_view(a: Claim, b: Claim) -> PairView:
    return PairView(
        stance=stance_relation(a, b),
        same_direction=a.polarity == b.polarity and a.polarity in (Polarity.POSITIVE, Polarity.NEGATIVE),
        same_subject=family(a.subject) is not None and family(a.subject) == family(b.subject),
        shared_object=bool(a.context.baselines and b.context.baselines
                           and family(a.context.baselines[0]) == family(b.context.baselines[0])),
        factors=compare_contexts(a.context, b.context, a.inherited_context, b.inherited_context),
    )


def _rid(a: str, b: str) -> str:
    return "rel_" + hashlib.sha1(f"{a}->{b}".encode()).hexdigest()[:12]


@dataclass
class PairCandidate:
    a: Claim  # older / target
    b: Claim  # newer / source
    similarity: float


ELIGIBLE = {ClaimType.FINDING, ClaimType.COMPARISON, ClaimType.LIMITATION, ClaimType.BACKGROUND}


def candidate_pairs(claims: list[Claim], papers: dict[str, Paper], models: ModelBundle, threshold: float,
                    max_pairs: int) -> list[PairCandidate]:
    eligible = [c for c in claims if c.claim_type in ELIGIBLE]
    if len(eligible) < 2:
        return []
    vecs = models.embedder.embed([c.normalized_text for c in eligible])
    sims = vecs @ vecs.T
    out: list[PairCandidate] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            if a.paper_id == b.paper_id:
                continue
            if a.claim_type == ClaimType.BACKGROUND and b.claim_type == ClaimType.BACKGROUND:
                continue
            sim = float(sims[i, j])
            shared_entity = bool({family(x) for x in a.context.models} & {family(x) for x in b.context.models})
            if sim < threshold and not (shared_entity and sim >= threshold * 0.9):
                continue
            ya, yb = papers[a.paper_id].year or 0, papers[b.paper_id].year or 0
            older, newer = (a, b) if (ya, a.claim_id) <= (yb, b.claim_id) else (b, a)
            out.append(PairCandidate(older, newer, sim))
    out.sort(key=lambda p: p.similarity, reverse=True)
    return out[:max_pairs]


def auto_threshold(models: ModelBundle, configured: float | None) -> float:
    if configured:
        return configured
    # bge-small cosine similarities sit high (~0.55 unrelated, ~0.8 related);
    # the hashing fallback is much sparser.
    return 0.72 if models.embedder.name == "fastembed" else 0.12


class PairAnalyzer:
    """Interface of the optional LLM pair analyser (implemented in :mod:`app.reasoning.alignment`)."""

    def analyze(self, pairs: list[PairCandidate], llm: BudgetedLLM) -> list[PairView | None]:  # pragma: no cover
        raise NotImplementedError


class RelationClassifier:
    def __init__(self, models: ModelBundle, llm: BudgetedLLM, pair_threshold: float | None = None,
                 config: RuleConfig | None = None, analyzer: PairAnalyzer | None = None, llm_reserve: int = 0):
        self.models = models
        self.llm = llm
        self.config = config or RuleConfig()
        self.analyzer = analyzer
        self.llm_reserve = llm_reserve  # LLM calls to leave unspent for later stages
        # similarity cut-offs are relative to the embedding backend's pairing threshold
        # (bge: t=0.72 → mid≈0.79, high≈0.86; hashing fallback: t=0.12 → mid≈0.34, high≈0.56)
        t = pair_threshold if pair_threshold is not None else auto_threshold(models, None)
        self.sim_low = t + 0.03
        self.sim_mid = t + (1 - t) * 0.25
        self.sim_high = t + (1 - t) * 0.5

    def nli_scores(self, pairs: list[PairCandidate]) -> list[tuple[dict[str, float], dict[str, float]]]:
        """NLI in both directions (A→B, B→A) for every pair."""
        nli_in: list[tuple[str, str]] = []
        for p in pairs:
            nli_in.append((p.a.normalized_text, p.b.normalized_text))
            nli_in.append((p.b.normalized_text, p.a.normalized_text))
        nli = self.models.nli.predict(nli_in) if nli_in else []
        return [(nli[2 * k], nli[2 * k + 1]) for k in range(len(pairs))]

    def views(self, pairs: list[PairCandidate], nli: list[tuple[dict[str, float], dict[str, float]]]) -> list[PairView]:
        """Stance + context comparison per pair: grounded LLM analysis when available, surface rules otherwise."""
        views = [rule_view(p.a, p.b) for p in pairs]
        if self.analyzer is not None and self.llm.enabled:
            # spend the LLM budget on likely conflicts first, then on the most similar pairs
            order = sorted(range(len(pairs)), key=lambda k: (-(nli[k][0]["contradiction"] + nli[k][1]["contradiction"]),
                                                             -pairs[k].similarity))
            order = order[: max(0, self.llm.remaining - self.llm_reserve)]
            analysed = self.analyzer.analyze([pairs[k] for k in order], self.llm)
            for k, view in zip(order, analysed):
                if view is not None:  # None = budget exhausted or provider failure → the rule-based view stands
                    views[k] = view
        return views

    def classify(self, pairs: list[PairCandidate]) -> list[Relationship]:
        if not pairs:
            return []
        nli = self.nli_scores(pairs)
        views = self.views(pairs, nli)
        rels: list[Relationship] = []
        for p, (ab, ba), view in zip(pairs, nli, views):
            rel = self.decide(p, ab, ba, view)
            if rel:
                rels.append(rel)
        return rels

    # ------------------------------------------------------------------ rules
    def decide(self, p: PairCandidate, ab: dict[str, float], ba: dict[str, float],
               view: PairView | None = None) -> Relationship | None:
        """Apply the decision rules to one pair. ``None`` means no relation is asserted."""
        a, b = p.a, p.b
        cfg = self.config
        view = view or rule_view(a, b)
        ent = max(ab["entailment"], ba["entailment"])
        con = (ab["contradiction"] + ba["contradiction"]) / 2
        neu = 1.0 - max(ent, con)
        stance_rel = view.stance if cfg.use_stance else None
        confident = view.confident and cfg.use_stance
        opposing = stance_rel == "oppose"
        factors = view.factors
        div = context_divergence(factors)
        signals = RelationSignals(similarity=round(p.similarity, 4), nli_entailment=round(ent, 4),
                                  nli_contradiction=round(con, 4), nli_neutral=round(neu, 4),
                                  polarity_conflict=opposing, context_divergence=round(div if div is not None else -1, 4))
        conflict_signals = signals.model_copy(update={"conflict_signal": True})
        differing = [f.factor for f in factors if f.status == "different"]
        # what the claims are *about* (models/baselines) is not experimental context
        # surface rules cannot tell whether a paper-level value applies to this claim, so they do not count it as
        # known context; the LLM analyser reports a value only where it judged that it does
        trusted = view.source == "llm"
        known = [f for f in factors if f.status != "unknown" and (trusted or not f.inherited) and f.factor not in _SUBJECT_FACTORS]
        base_conf = min(a.confidence, b.confidence)
        background_involved = ClaimType.BACKGROUND in (a.claim_type, b.claim_type)
        same_subject = view.same_subject and cfg.use_stance

        # a semantic stance judgement that the claims answer different questions ends the analysis
        if cfg.use_stance and view.comparable is False:
            return None

        # ---- 1. disagreement -------------------------------------------------------
        # NLI alone must be confident, and never overrides two claims that report the same direction of
        # effect ("X improves Y" vs "Z improves Y" is not a contradiction, although NLI models often say so).
        # A surface-cue stance needs some NLI corroboration; a semantic (LLM) stance does not.
        same_direction = view.same_direction and cfg.use_stance
        contradiction_signal = (con >= 0.7 and stance_rel != "agree" and not same_direction) or \
            (opposing and (confident or con >= 0.2 or (ent < 0.2 and p.similarity >= self.sim_mid)))
        if contradiction_signal:
            strength = max(con, 0.6 if opposing else 0.0)
            if not cfg.use_context_gate:
                relation, uncertain = RelationType.CONTRADICTS, False
                reasoning = (f"The claims appear to disagree (NLI contradiction {con:.2f}"
                             f"{', opposing stance' if opposing else ''}). Context was not consulted.")
                conf = 0.6 * strength + 0.4 * base_conf
            elif div is None or not known:
                relation, uncertain = RelationType.CONTRADICTS, True
                reasoning = ("The claims appear to disagree (NLI contradiction "
                             f"{con:.2f}{', opposing stance' if opposing else ''}), but their own passages state almost no "
                             "experimental context, so ClaimGraph cannot tell whether this is a real contradiction.")
                conf = 0.35 * strength + 0.15 * base_conf
            elif differing and (div >= 0.3 or trusted):
                # lexicon-matched factors are noisy, so they must add up (divergence >= 0.3); a semantically judged
                # table already lists only differences that could explain the findings, so one is enough
                relation, uncertain = RelationType.QUALIFIES, False
                reasoning = (f"The claims point in opposite directions (NLI contradiction {con:.2f}"
                             f"{', opposing stance' if opposing else ''}) but were made under different "
                             f"{', '.join(differing[:4])}. The later claim therefore qualifies the earlier one: "
                             "the finding appears to depend on these conditions rather than being refuted.")
                conf = 0.5 * strength + 0.3 * base_conf + 0.2 * min(1.0, div)
            else:
                relation, uncertain = RelationType.CONTRADICTS, False
                shared = [f.factor for f in factors if f.status in ("same", "overlap")]
                reasoning = (f"The claims report opposing findings (NLI contradiction {con:.2f}"
                             f"{', opposing stance' if opposing else ''}) under substantially similar conditions "
                             f"(shared: {', '.join(shared[:4]) or 'n/a'}).")
                conf = 0.6 * strength + 0.4 * base_conf
            if background_involved:
                uncertain = True
                conf *= 0.75
                reasoning += " One side is a belief attributed to prior work rather than a new result."
            return self._rel(b, a, relation, conf, reasoning, conflict_signals, factors, uncertain, view)

        # ---- 2. agreement (NLI entailment, or the same directional stance) ----------
        shared_object = view.shared_object and cfg.use_stance
        stance_agree = stance_rel == "agree" and (confident or (con < 0.3 and (
            (shared_object and p.similarity >= self.sim_low) or p.similarity >= self.sim_mid)))
        agree = ent >= 0.5 or (ent >= 0.3 and p.similarity >= self.sim_high) or stance_agree
        if agree:
            basis = (f"NLI entailment {ent:.2f}" if ent >= 0.3 else
                     f"both report the same direction of effect for {a.subject} (similarity {p.similarity:.2f})")
            strength = max(ent, 0.55 if stance_rel == "agree" else 0.0)
            new_values = [f.factor for f in factors if f.status == "different" and (trusted or not f.inherited)
                          and f.factor in ("dataset", "data scale", "task", "population", "experimental conditions")]
            if new_values and b.claim_type != ClaimType.BACKGROUND:
                reasoning = (f"The later claim is consistent with the earlier one ({basis}) and extends it to a "
                             f"different {', '.join(new_values[:3])}.")
                return self._rel(b, a, RelationType.EXTENDS, 0.45 * strength + 0.35 * base_conf + 0.1, reasoning, signals,
                                 factors, background_involved, view)
            reasoning = f"Both claims assert compatible findings ({basis})."
            if div is not None and div < 0.3:
                reasoning += " Their reported conditions also align."
            return self._rel(b, a, RelationType.SUPPORTS, 0.55 * strength + 0.35 * base_conf + 0.1, reasoning, signals,
                             factors, background_involved, view)

        # ---- 3. weaker structural relations (always flagged uncertain) --------------
        # a surface-cue parity mismatch is only believed for very similar sentences; a semantic (LLM) "partial"
        # stance is treated like "agree" and "oppose": it needs no corroboration
        if stance_rel == "parity_mismatch" and (confident or p.similarity >= self.sim_mid):
            reasoning = ("One claim narrows, weakens or conditions the other on the same relationship; the claims "
                         "neither simply agree nor oppose." if confident else
                         "One claim reports parity while the other reports a clear advantage for the same comparison; "
                         "the later claim narrows the strength of the earlier finding.")
            return self._rel(b, a, RelationType.QUALIFIES, 0.35 + 0.3 * base_conf, reasoning, signals, factors,
                             not confident, view)

        if same_subject and p.similarity >= self.sim_high and neu >= 0.7 and not background_involved:
            added = [f.factor for f in factors if f.status == "different" and not f.inherited
                     and f.factor in ("training & design", "experimental conditions", "dataset")]
            if added:
                reasoning = (f"The claims address the same question about {a.subject} (similarity {p.similarity:.2f}) "
                             f"with no logical conflict; the later one adds a different {', '.join(added[:2])}.")
                return self._rel(b, a, RelationType.EXTENDS, 0.3 + 0.3 * base_conf, reasoning, signals, factors, True, view)
        return None

    @staticmethod
    def _rel(src: Claim, tgt: Claim, relation: RelationType, conf: float, reasoning: str, signals: RelationSignals,
             factors: list[FactorComparison], uncertain: bool, view: PairView | None = None) -> Relationship:
        from_llm = view is not None and view.source == "llm"
        return Relationship(
            relation_id=_rid(src.claim_id, tgt.claim_id),
            source_claim_id=src.claim_id,
            target_claim_id=tgt.claim_id,
            relation=relation,
            confidence=round(float(np.clip(conf, 0.05, 0.95)), 3),
            reasoning=reasoning,
            signals=signals,
            factors=factors,
            uncertain=uncertain,
            decided_by="rules+llm" if from_llm else "rules",
            llm_note=view.note if from_llm else None,
        )
