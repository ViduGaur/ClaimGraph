"""Experiment core: per-topic artefacts and the systems compared in the paper.

For each topic the harness produces, once:

* ``claims.json``  — papers and extracted claims (shared by every system);
* ``pairs.jsonl``  — every candidate pair with its raw signals: similarity, NLI in both
  directions and, when an LLM is configured, the raw (cached) LLM outputs.

Every system — baselines, ablations and the full hybrid — is then a *pure function* of a
pair record, so all systems are compared on exactly the same pairs and re-scoring is offline.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.evaluation import EVAL_DIR, run_dir
from app.evaluation.corpus import Topic, get_topic, load_corpus
from app.evaluation.llm_baseline import DIRECT_SCHEMA, DIRECT_SYSTEM, direct_label, direct_prompt
from app.evaluation.metrics import ABSTAIN, PRED_LABELS
from app.extraction.claims import ClaimExtractor
from app.providers.llm import BudgetedLLM
from app.providers.models import ModelBundle
from app.reasoning import alignment, comparability
from app.reasoning.conflicts import analyze_conflicts
from app.reasoning.relations import (
    PairCandidate,
    PairView,
    RelationClassifier,
    RuleConfig,
    auto_threshold,
    candidate_pairs,
    rule_view,
)
from app.schemas.domain import Claim, ConflictVerdict, Paper, Relationship, RelationType

log = logging.getLogger(__name__)

# Two levels (see app.evaluation.metrics): a relation, and for face-value conflicts a verdict. A system that sees a
# conflict but will not assert it (relation CONTRADICTS flagged uncertain, verdict insufficient) is scored as ABSTAIN.
LABELS = PRED_LABELS
VERDICT_OF_LABEL = {"CONTRADICTS": "genuine", "QUALIFIES": "context_dependent", ABSTAIN: "insufficient"}


def two_level(label: str | None, verdict: str | None = None) -> tuple[str | None, str | None]:
    """(relation, conflict_verdict) as reported: an abstention is an uncertain CONTRADICTS with verdict insufficient."""
    if label is None:
        return None, None
    return ("CONTRADICTS" if label == ABSTAIN else label), verdict


def pred_verdict(label: str | None, rel: Relationship | None, *, rule_based: bool) -> str | None:
    """The conflict verdict a prediction implies, or None when the system saw no face-value conflict.

    Rule-based systems carry an explicit conflict signal. For systems that only output a label (NLI, direct LLM),
    CONTRADICTS is read as a genuine conflict, QUALIFIES as a context-dependent one and an abstention as insufficient.
    """
    if label is None:
        return None
    if rule_based and (rel is None or not rel.signals.conflict_signal):
        return None
    return VERDICT_OF_LABEL.get(label)
NEUTRAL_NLI = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}


# --------------------------------------------------------------------------- artefacts
@dataclass
class TopicData:
    topic: Topic
    papers: dict[str, Paper]
    claims: dict[str, Claim]
    records: list[dict]  # one per candidate pair
    threshold: float
    meta: dict = field(default_factory=dict)

    def pair(self, rec: dict) -> PairCandidate:
        return PairCandidate(self.claims[rec["a"]], self.claims[rec["b"]], rec["similarity"])


def pair_id(a: str, b: str) -> str:
    return "pair_" + hashlib.sha1(f"{a}|{b}".encode()).hexdigest()[:12]


def topic_dir(run: str, topic_id: str) -> Path:
    return run_dir(run) / topic_id


def prepare_topic(topic_id: str, run: str, settings: Settings, models: ModelBundle, *, max_claims: int,
                  max_pairs: int, llm: BudgetedLLM | None = None) -> TopicData:
    """Extract claims and candidate pairs (+ similarity and NLI) for one frozen corpus and save them.

    The claim set is shared by every compared system. With ``llm`` it is the cue-based candidates
    screened by the LLM (verbatim sentences either way); without, the cue ranking alone.
    """
    topic = get_topic(topic_id)
    docs = load_corpus(topic_id)
    no_llm = BudgetedLLM(None, 0)
    extractor = ClaimExtractor(models, llm or no_llm, max_claims, False, llm_selection=llm is not None)
    qvec = models.embedder.embed([topic.question])[0]
    claims: list[Claim] = []
    for doc in docs:
        found, _ = extractor.extract(topic.question, qvec, doc)
        claims.extend(found)
    papers = {d.paper.paper_id: d.paper for d in docs}
    threshold = auto_threshold(models, settings.pair_similarity_threshold)
    pairs = candidate_pairs(claims, papers, models, threshold, max_pairs)
    classifier = RelationClassifier(models, no_llm, pair_threshold=threshold)
    nli = classifier.nli_scores(pairs)
    records = [{"pair_id": pair_id(p.a.claim_id, p.b.claim_id), "topic": topic_id, "a": p.a.claim_id, "b": p.b.claim_id,
                "similarity": round(p.similarity, 4), "nli_ab": _round(ab), "nli_ba": _round(ba)}
               for p, (ab, ba) in zip(pairs, nli)]
    data = TopicData(topic, papers, {c.claim_id: c for c in claims}, records, threshold,
                     meta={"embedding": models.embedder.name, "nli": models.nli.name, "max_claims": max_claims,
                           "claim_selection": "llm" if llm is not None else "rules",
                           "max_pairs": max_pairs, "n_papers": len(docs),
                           "full_text": sum(1 for d in docs if d.paper.text_source == "full_text")})
    save_topic(data, run)
    return data


def _round(d: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in d.items()}


def save_topic(data: TopicData, run: str) -> None:
    out = topic_dir(run, data.topic.id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "claims.json").write_text(json.dumps({
        "topic": data.topic.__dict__, "threshold": data.threshold, "meta": data.meta,
        "papers": [p.model_dump() for p in data.papers.values()],
        "claims": [c.model_dump(mode="json") for c in data.claims.values()],
    }, ensure_ascii=False, indent=1))
    with (out / "pairs.jsonl").open("w") as fh:
        for rec in data.records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_topic(run: str, topic_id: str) -> TopicData:
    out = topic_dir(run, topic_id)
    raw = json.loads((out / "claims.json").read_text())
    records = [json.loads(line) for line in (out / "pairs.jsonl").read_text().splitlines() if line.strip()]
    t = raw["topic"]
    return TopicData(Topic(t["id"], t["split"], t["question"], t.get("seeds", [])),
                     {p["paper_id"]: Paper(**p) for p in raw["papers"]},
                     {c["claim_id"]: Claim(**c) for c in raw["claims"]}, records, raw["threshold"], raw.get("meta", {}))


# --------------------------------------------------------------------------- LLM signals
LLM_FIELDS = ("llm_pair", "llm_direct_claims", "llm_direct_context")


def add_llm_signals(data: TopicData, llm: BudgetedLLM, *, concurrency: int = 6, fields: tuple[str, ...] = LLM_FIELDS,
                    only: set[str] | None = None, progress: Callable[[str], None] | None = None) -> dict[str, int]:
    """Fill the raw LLM outputs of each record (cached on disk, so this is resumable and replayable)."""
    todo = [(rec, f) for rec in data.records if only is None or rec["pair_id"] in only for f in fields if not rec.get(f)]
    failed = {f: 0 for f in fields}

    def call(item: tuple[dict, str]) -> None:
        rec, name = item
        a, b = data.claims[rec["a"]], data.claims[rec["b"]]
        if name == "llm_pair":
            out = llm.complete_json(alignment.SYSTEM, alignment.render_pair(a, b, data.papers), alignment.SCHEMA, max_tokens=alignment.MAX_TOKENS)
        else:
            variant = "claims" if name == "llm_direct_claims" else "context"
            out = llm.complete_json(DIRECT_SYSTEM, direct_prompt(a, b, data.papers, variant), DIRECT_SCHEMA, max_tokens=160)
        if out:
            rec[name] = out
        else:
            failed[name] += 1

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for i, _ in enumerate(pool.map(call, todo), 1):
            if progress and i % 50 == 0:
                progress(f"{data.topic.id}: {i}/{len(todo)} LLM calls")
    return failed


# --------------------------------------------------------------------------- systems
@dataclass(frozen=True)
class SystemSpec:
    name: str
    title: str  # as it appears in tables
    group: str  # 'baseline' | 'rules' | 'hybrid' | 'llm'
    needs: tuple[str, ...] = ()  # LLM record fields the system depends on
    description: str = ""


def final_label(rel: Relationship | None, cfg: RuleConfig, claims: dict[str, Claim]) -> str:
    """Map a relationship (or none) onto a prediction label: one of the five relations, or ABSTAIN."""
    if rel is None:
        return "UNRELATED"
    if rel.relation == RelationType.CONTRADICTS and not cfg.use_context_gate:
        return "CONTRADICTS"
    conflicts = analyze_conflicts([rel], claims)
    if not conflicts:
        return rel.relation.value
    return {ConflictVerdict.GENUINE: "CONTRADICTS", ConflictVerdict.CONTEXT_DEPENDENT: "QUALIFIES",
            ConflictVerdict.INSUFFICIENT_EVIDENCE: ABSTAIN}[conflicts[0].verdict]


def llm_view(data: TopicData, rec: dict, *, enforce_grounding: bool = True,
             stats: alignment.GroundingStats | None = None) -> PairView | None:
    raw = rec.get("llm_pair")
    if not raw:
        return None
    return alignment.view_from_output(raw, data.claims[rec["a"]], data.claims[rec["b"]], data.papers, stats, enforce_grounding)


class Systems:
    """All compared systems as functions ``record -> (label, relationship | None)``."""

    def __init__(self, data: TopicData):
        self.data = data
        self._clf: dict[RuleConfig, RelationClassifier] = {}

    def _classifier(self, cfg: RuleConfig) -> RelationClassifier:
        if cfg not in self._clf:
            self._clf[cfg] = RelationClassifier(None, BudgetedLLM(None, 0), pair_threshold=self.data.threshold, config=cfg)  # type: ignore[arg-type]
        return self._clf[cfg]

    def _rules(self, rec: dict, cfg: RuleConfig, view: PairView | None, *, use_nli: bool = True) -> tuple[str, Relationship | None]:
        pair = self.data.pair(rec)
        ab, ba = (rec["nli_ab"], rec["nli_ba"]) if use_nli else (NEUTRAL_NLI, NEUTRAL_NLI)
        rel = self._classifier(cfg).decide(pair, ab, ba, view or rule_view(pair.a, pair.b))
        return final_label(rel, cfg, self.data.claims), rel

    # ---- baselines
    def similarity(self, rec: dict) -> tuple[str, None]:
        return "SUPPORTS", None  # every candidate pair is above the similarity threshold

    def nli_only(self, rec: dict, threshold: float = 0.5) -> tuple[str, None]:
        con = (rec["nli_ab"]["contradiction"] + rec["nli_ba"]["contradiction"]) / 2
        ent = max(rec["nli_ab"]["entailment"], rec["nli_ba"]["entailment"])
        if con >= threshold and con >= ent:
            return "CONTRADICTS", None
        return ("SUPPORTS" if ent >= threshold else "UNRELATED"), None

    def llm_direct(self, rec: dict, variant: str) -> tuple[str | None, None]:
        label = direct_label(rec.get(f"llm_direct_{variant}"))
        return (ABSTAIN if label == "INSUFFICIENT" else label), None

    def constant(self, rec: dict, label: str) -> tuple[str, None]:
        return label, None

    # ---- rule-based ClaimGraph (no LLM)
    def rules(self, rec: dict, cfg: RuleConfig = RuleConfig()) -> tuple[str, Relationship | None]:
        return self._rules(rec, cfg, None)

    # ---- hybrid ClaimGraph (LLM-analysed stance + grounded context, rule decision)
    def hybrid(self, rec: dict, cfg: RuleConfig = RuleConfig(), *, enforce_grounding: bool = True, use_nli: bool = True,
               gate_always: str | None = None) -> tuple[str | None, Relationship | None]:
        """``gate_always`` replaces the context gate by a constant answer for every face-value conflict."""
        view = llm_view(self.data, rec, enforce_grounding=enforce_grounding)
        if view is None:
            return None, None
        label, rel = self._rules(rec, cfg, view, use_nli=use_nli)
        if gate_always and rel is not None and rel.signals.conflict_signal:
            label = gate_always
        return label, rel


    # ---- V2: the gate replaced by the comparability model (frozen, development-calibrated parameters)
    def v2(self, rec: dict, params: comparability.ComparabilityParams) -> tuple[str | None, Relationship | None]:
        """Stance and the factor table are V1's; only what happens to a face-value conflict differs."""
        label, rel = self.hybrid(rec)
        if rel is None or not rel.signals.conflict_signal:
            return label, rel
        status = comparability.assess(rel.factors, params).status
        return {"genuine": "CONTRADICTS", "context_dependent": "QUALIFIES", "insufficient": ABSTAIN}[status], rel


V2_PARAMS_PATH = EVAL_DIR / "v2_params.json"
V2_PARAMS = comparability.load_params(V2_PARAMS_PATH)
NO_GATE = RuleConfig(use_context_gate=False)
NO_STANCE = RuleConfig(use_stance=False)

SYSTEM_SPECS: list[SystemSpec] = [
    SystemSpec("similarity", "Similarity only", "baseline", description="every candidate pair is SUPPORTS"),
    SystemSpec("nli", "NLI only", "baseline", description="bidirectional NLI, threshold 0.5"),
    SystemSpec("llm_direct_claims", "LLM direct (claims only)", "llm", ("llm_direct_claims",)),
    SystemSpec("llm_direct_context", "LLM direct (+ passages)", "llm", ("llm_direct_context",)),
    SystemSpec("rules_full", "ClaimGraph-R (rules, lexicon context)", "rules", description="no LLM"),
    SystemSpec("rules_no_gate", "ClaimGraph-R w/o context gate", "rules"),
    SystemSpec("rules_no_stance", "ClaimGraph-R w/o stance", "rules"),
    SystemSpec("hybrid_full", "ClaimGraph (full)", "hybrid", ("llm_pair",)),
    SystemSpec("hybrid_no_gate", "  w/o context gate", "hybrid", ("llm_pair",)),
    SystemSpec("hybrid_no_stance", "  w/o stance", "hybrid", ("llm_pair",)),
    SystemSpec("hybrid_no_nli", "  w/o NLI", "hybrid", ("llm_pair",)),
    SystemSpec("hybrid_no_grounding", "  w/o quote grounding", "hybrid", ("llm_pair",)),
    SystemSpec("hybrid_always_qualify", "  gate: always context-dependent", "hybrid", ("llm_pair",),
               description="degenerate gate: every face-value conflict becomes QUALIFIES"),
    SystemSpec("hybrid_always_abstain", "  gate: always abstain", "hybrid", ("llm_pair",),
               description="degenerate gate: the system abstains on every face-value conflict"),
    SystemSpec("always_unrelated", "Constant: UNRELATED", "degenerate", description="never relates any pair"),
    SystemSpec("always_qualifies", "Constant: QUALIFIES", "degenerate", description="calls every pair QUALIFIES"),
]
SYSTEM_SPECS += [SystemSpec(f"v2_{name}", f"V2 comparability model: {name}", "hybrid", ("llm_pair",),
                            description="face-value conflicts decided by the development-calibrated comparability model")
                 for name in V2_PARAMS]
V2_SYSTEMS = [f"v2_{name}" for name in V2_PARAMS]
RULE_BASED = {s.name for s in SYSTEM_SPECS if s.group in ("rules", "hybrid")}
SPEC_BY_NAME = {s.name: s for s in SYSTEM_SPECS}


def predict(systems: Systems, name: str, rec: dict) -> tuple[str | None, Relationship | None]:
    """Label of system ``name`` for one pair record; ``None`` when its LLM signal is missing."""
    s = systems
    table: dict[str, Callable[[], tuple[str | None, Relationship | None]]] = {
        "similarity": lambda: s.similarity(rec),
        "nli": lambda: s.nli_only(rec),
        "llm_direct_claims": lambda: s.llm_direct(rec, "claims"),
        "llm_direct_context": lambda: s.llm_direct(rec, "context"),
        "rules_full": lambda: s.rules(rec),
        "rules_no_gate": lambda: s.rules(rec, NO_GATE),
        "rules_no_stance": lambda: s.rules(rec, NO_STANCE),
        "hybrid_full": lambda: s.hybrid(rec),
        "hybrid_no_gate": lambda: s.hybrid(rec, NO_GATE),
        "hybrid_no_stance": lambda: s.hybrid(rec, NO_STANCE),
        "hybrid_no_nli": lambda: s.hybrid(rec, use_nli=False),
        "hybrid_no_grounding": lambda: s.hybrid(rec, enforce_grounding=False),
        "hybrid_always_qualify": lambda: s.hybrid(rec, gate_always="QUALIFIES"),
        "hybrid_always_abstain": lambda: s.hybrid(rec, gate_always=ABSTAIN),
        "always_unrelated": lambda: s.constant(rec, "UNRELATED"),
        "always_qualifies": lambda: s.constant(rec, "QUALIFIES"),
        **{f"v2_{n}": (lambda q=q: s.v2(rec, q)) for n, q in V2_PARAMS.items()},
    }
    return table[name]()


def predict_full(systems: Systems, name: str, rec: dict) -> tuple[str | None, str | None, Relationship | None]:
    """(label, conflict verdict, relationship) of system ``name`` for one pair record."""
    label, rel = predict(systems, name, rec)
    return label, pred_verdict(label, rel, rule_based=name in RULE_BASED), rel


def predict_all(data: TopicData, names: list[str] | None = None) -> dict[str, dict[str, str | None]]:
    """{pair_id: {system: label}} for every record of a topic."""
    systems = Systems(data)
    names = names or [s.name for s in SYSTEM_SPECS]
    return {rec["pair_id"]: {n: predict(systems, n, rec)[0] for n in names} for rec in data.records}


# --------------------------------------------------------------------------- grounded explanations
def add_explanations(data: TopicData, llm: BudgetedLLM, *, concurrency: int = 6, system: str = "hybrid_full",
                     refresh: bool = False) -> dict[str, int]:
    """Explain every face-value conflict of ``system`` (default: the full system) and store the record on the pair.

    Runs after the decision and cannot change it: the explanation model only sees the decision's own evidence.
    """
    from app.reasoning.explanations import explain_one, explanation_artifact

    systems = Systems(data)
    todo = []
    for rec in data.records:
        if refresh:
            rec.pop("explanation", None)
        label, rel = predict(systems, system, rec)
        if rel is None or not rel.signals.conflict_signal:
            rec.pop("explanation", None)  # no longer a conflict under the current rules
            continue
        if not rec.get("explanation"):
            todo.append((rec, rel))

    def run(item: tuple[dict, Relationship]) -> None:
        rec, rel = item
        a, b = data.claims[rec["a"]], data.claims[rec["b"]]
        conflict = analyze_conflicts([rel], data.claims)[0]
        conflict = explain_one(conflict, rel, a, b, llm)
        rec["explanation"] = explanation_artifact(conflict, rel, a, b)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(run, todo))
    done = [r["explanation"] for r in data.records if r.get("explanation")]
    return {"conflicts": len(done), "llm": sum(1 for x in done if x["explanation_source"] == "llm"),
            "template": sum(1 for x in done if x["explanation_source"] == "template"), "new": len(todo)}
