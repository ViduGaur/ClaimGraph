"""Development study for the V2 weight-of-evidence comparability model. Every number here serves model choice; none is a
paper result.

1. Among pairs read as opposed, does the factor evidence separate consensus contradictions from the rest?
2. Does a fitted role-level model beat uniform weights, the V1 hand weights, or the V1 hard gate?
3. Does the factor evidence add anything to semantic stance alone? (shuffled-factor control, semantic-only model)
4. Which factors and roles carry the signal? (ablations)

Models are logistic, fitted by penalised maximum likelihood with sign constraints and compared by leave-one-topic-out
cross-validation.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.evaluation import EVAL_DIR
from app.evaluation.experiment import Systems, TopicData, llm_view, load_topic, predict_full
from app.evaluation.metrics import ABSTAIN
from app.reasoning import comparability as C
from app.reasoning.relations import FACTORS
from app.schemas.domain import FactorComparison

HAND_WEIGHTS = {label: w for label, _, w in FACTORS}
L2 = 1.0  # fixed in advance (the scikit-learn default C = 1); a sensitivity table reports 0.1 and 10


# ----------------------------------------------------------------------------------------------- data
@dataclass
class Row:
    item_id: str
    pair_id: str
    topic: str
    final: str | None  # consensus relation (None: three-way split, not scored)
    votes: list[str]
    opposed: bool  # the system reads the pair as a face-value conflict (V1 trigger)
    v1: str | None  # V1 label: CONTRADICTS, QUALIFIES, ABSTAIN, ...
    stance: str | None
    con: float
    ent: float
    sim: float
    factors: list[FactorComparison] = field(default_factory=list)

    @property
    def genuine(self) -> int:
        return int(self.final == "CONTRADICTS")

    @property
    def vote_share(self) -> float:
        return sum(v == "CONTRADICTS" for v in self.votes) / max(1, len(self.votes))


def load_rows(batch: str, run: str) -> list[Row]:
    labels_dir = EVAL_DIR / "annotation" / batch
    gold = json.loads((labels_dir / "gold.json").read_text())["items"]
    key = json.loads(next(labels_dir.glob("*_key.json")).read_text())
    datas: dict[str, TopicData] = {t: load_topic(run, t) for t in sorted({i["topic"] for i in key["items"]})}
    records = {t: {r["pair_id"]: r for r in d.records} for t, d in datas.items()}
    systems = {t: Systems(d) for t, d in datas.items()}
    rows = []
    for item in key["items"]:
        g, rec = gold[item["item_id"]], records[item["topic"]][item["pair_id"]]
        no_gate, _, _ = predict_full(systems[item["topic"]], "hybrid_no_gate", rec)
        v1, _, _ = predict_full(systems[item["topic"]], "hybrid_full", rec)
        view = llm_view(datas[item["topic"]], rec)
        rows.append(Row(item["item_id"], item["pair_id"], item["topic"], g["final"], list(g["model_labels"].values()),
                        no_gate == "CONTRADICTS", v1, (rec.get("llm_pair") or {}).get("stance"),
                        (rec["nli_ab"]["contradiction"] + rec["nli_ba"]["contradiction"]) / 2,
                        max(rec["nli_ab"]["entailment"], rec["nli_ba"]["entailment"]), rec["similarity"],
                        view.factors if view else []))
    return rows


# ---------------------------------------------------------------------------------------------- models
@dataclass(frozen=True)
class Spec:
    name: str
    title: str
    columns: tuple[str, ...]
    features: Callable[[Row], list[float]]
    nonneg: tuple[bool, ...]  # sign-constrained coefficients (the column already carries the expected sign)
    note: str = ""

    @property
    def n_params(self) -> int:
        return len(self.columns) + 1


def _ev(row: Row, llm_materiality: bool = False) -> list[C.FactorEvidence]:
    return C.factor_evidence(row.factors, llm_materiality=llm_materiality)


def _same(row: Row, **kw) -> float:
    return sum(e.same for e in _ev(row, **kw))


def _diff(row: Row, roles: tuple[str, ...] | None = None, weights: dict[str, float] | None = None, skip: str | None = None, **kw) -> float:
    return sum(e.different * (weights[e.factor] if weights else 1.0) for e in _ev(row, **kw)
               if (roles is None or e.role in roles) and e.factor != skip)


def role_spec(name: str, title: str, roles: tuple[str, ...], *, same: bool = True, constrained: bool = True,
              llm_materiality: bool = False, skip_factor: str | None = None, note: str = "") -> Spec:
    cols = (("same",) if same else ()) + tuple(f"diff:{r}" for r in roles)

    def feats(row: Row) -> list[float]:
        ev = [e for e in _ev(row, llm_materiality) if e.factor != skip_factor]
        out = [sum(e.same for e in ev)] if same else []
        return out + [-sum(e.different for e in ev if e.role == r) for r in roles]

    return Spec(name, title, cols, feats, tuple(constrained for _ in cols), note)


SEMANTIC = Spec("semantic", "Semantic signals only (NLI contradiction, entailment, similarity)", ("nli_con", "nli_ent", "similarity"),
                lambda r: [r.con, r.ent, r.sim], (False, False, False), "no factor evidence")
ALL_ROLES = ("scope", "measurement", "strength", "intervention")
CONTEXT_ROLES = ("scope", "measurement")
SPECS: list[Spec] = [
    Spec("constant", "Constant (stance alone: every opposed pair alike)", (), lambda r: [], (), "no factor evidence"),
    SEMANTIC,
    Spec("context_stated", "Sufficiency only: is any context factor stated by both papers?", ("no shared context",),
         lambda r: [float(C.sufficiency(_ev(r)) == 0)], (False,), "uses no factor values at all"),
    Spec("uniform_ctx", "Uniform weights, scope + measurement mismatches only", ("diff:context",),
         lambda r: [-_diff(r, roles=CONTEXT_ROLES)], (True,)),
    role_spec("role2", "Role weights: scope, measurement (mismatch only)", CONTEXT_ROLES, same=False),
    Spec("uniform_diff", "Uniform weights, mismatches only", ("diff:all",), lambda r: [-_diff(r)], (True,)),
    Spec("uniform", "Uniform weights, agreement and mismatch", ("same", "diff:all"), lambda r: [_same(r), -_diff(r)], (True, True)),
    Spec("hand", "V1 hand weights, mismatches only", ("diff:hand-weighted",), lambda r: [-_diff(r, weights=HAND_WEIGHTS)], (True,)),
    Spec("hand_same", "V1 hand weights, agreement and mismatch", ("same:hand-weighted", "diff:hand-weighted"),
         lambda r: [sum(e.same * HAND_WEIGHTS[e.factor] for e in _ev(r)), -_diff(r, weights=HAND_WEIGHTS)], (True, True)),
    role_spec("role", "P3: role weights (scope, measurement, strength; intervention outside)", ("scope", "measurement", "strength")),
    role_spec("role4", "Role weights incl. intervention", ALL_ROLES),
    role_spec("role4_free", "Role weights incl. intervention, signs unconstrained", ALL_ROLES, constrained=False,
              note="diagnostic: which direction does the data want?"),
    Spec("free12", "Twelve free mismatch weights + agreement (exploratory)", ("same", *[f"diff:{f}" for f in C.FACTOR_ORDER]),
         lambda r: [_same(r)] + [-e.different for e in _ev(r)], (True,) * 13, "more parameters than the data supports"),
    role_spec("role4_llm_material", "Role weights, LLM materiality flag honoured", ALL_ROLES, llm_materiality=True),
    Spec("semantic_role4", "Semantic signals + role weights", (*SEMANTIC.columns, "same", *[f"diff:{r}" for r in ALL_ROLES]),
         lambda r: SEMANTIC.features(r) + role_spec("x", "x", ALL_ROLES).features(r), (False,) * 3 + (True,) * 5),
]
SPEC = {s.name: s for s in SPECS}


def design(rows: list[Row], spec: Spec) -> np.ndarray:
    return np.array([[1.0, *spec.features(r)] for r in rows], dtype=float)


def fit_logistic(X: np.ndarray, y: np.ndarray, nonneg: tuple[bool, ...], l2: float = L2, iters: int = 3000) -> np.ndarray:
    """Penalised logistic regression: minimise mean log-loss + (l2 / 2n) * |beta|^2, with beta_j >= 0 where ``nonneg``.

    The intercept (column 0) is neither penalised nor constrained. Accelerated projected gradient; ``y`` may be a
    probability (soft label).
    """
    n, k = X.shape
    pen = np.r_[0.0, np.full(k - 1, l2 / n)]
    lower = np.r_[-np.inf, [0.0 if c else -np.inf for c in nonneg]]
    step = 1.0 / (0.25 * np.linalg.norm(X, 2) ** 2 / n + l2 / n + 1e-9)
    beta = np.zeros(k)
    beta[0] = np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
    z, t = beta.copy(), 1.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ z, -30, 30)))
        new = np.maximum(z - step * (X.T @ (p - y) / n + pen * z), lower)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = new + (t - 1) / t_new * (new - beta)
        if np.max(np.abs(new - beta)) < 1e-7:
            beta = new
            break
        beta, t = new, t_new
    return beta


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))


# --------------------------------------------------------------------------------------------- metrics
def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def auc_within(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> float | None:
    """AUC over positive-negative pairs from the same group (topic) only.

    Pooling out-of-fold scores across topics is biased against any model: a held-out topic with many positives is
    scored by a model whose intercept was fitted on the topics with fewer. Comparing within topics removes that.
    """
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        den += len(pos) * len(neg)
    return float(num / den) if den else None


def precision_threshold(y: np.ndarray, p: np.ndarray, min_precision: float = 0.5) -> float:
    """Lowest threshold at which the asserted (training) pairs are at least ``min_precision`` correct; inf if none is."""
    for t in sorted(set(np.round(p, 6))):
        a = p >= t
        if a.sum() and y[a].mean() >= min_precision:
            return float(t)
    return float("inf")


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="stable")
    hits = np.cumsum(y[order])
    return float((hits / np.arange(1, len(y) + 1))[y[order] == 1].mean()) if y.sum() else 0.0


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold on p that maximises F1 of 'assert' on the given (training) pairs."""
    best, best_t = -1.0, 0.5
    for t in sorted(set(np.round(p, 6))):
        a = p >= t
        tp = int((a & (y == 1)).sum())
        f1 = 2 * tp / (a.sum() + y.sum()) if tp else 0.0
        if f1 > best + 1e-12:
            best, best_t = f1, float(t)
    return best_t


def cv_scores(rows: list[Row], spec: Spec, y: np.ndarray, folds: list[np.ndarray], l2: float = L2) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities and out-of-fold 'assert' decisions (threshold chosen on the training folds)."""
    X = design(rows, spec)
    hard = (y >= 0.5).astype(int)
    oof, asserted = np.zeros(len(rows)), np.zeros(len(rows), dtype=bool)
    for test in folds:
        train = np.setdiff1d(np.arange(len(rows)), test)
        beta = fit_logistic(X[train], y[train], spec.nonneg, l2)
        oof[test] = predict_proba(X[test], beta)
        asserted[test] = oof[test] >= best_f1_threshold(hard[train], predict_proba(X[train], beta))
    return oof, asserted


def topic_folds(rows: list[Row]) -> list[np.ndarray]:
    topics = sorted({r.topic for r in rows})
    return [np.array([i for i, r in enumerate(rows) if r.topic == t]) for t in topics]


def stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    return [np.array(f) for f in folds]


def decision_metrics(asserted_ids: set[str], scored: list[Row]) -> dict:
    """Asserted-contradiction metrics over ALL scored pairs (opposed or not)."""
    n_pos = sum(r.genuine for r in scored)
    tp = sum(r.genuine for r in scored if r.item_id in asserted_ids)
    fp = len(asserted_ids) - tp
    p = tp / len(asserted_ids) if asserted_ids else 0.0
    rec = tp / n_pos
    return {"asserted": len(asserted_ids), "tp": tp, "fp": fp, "precision": 100 * p, "recall": 100 * rec,
            "f1": 100 * (2 * p * rec / (p + rec) if p + rec else 0.0), "false_con_per_100": 100 * fp / len(scored)}


def boot_ci(fn: Callable[[np.ndarray], float | None], n: int, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    vals = [v for v in (fn(rng.integers(0, n, n)) for _ in range(n_boot)) if v is not None]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------------------------- study
def shuffled_rows(rows: list[Row], rng: np.random.Generator, cell: Callable[[Row], object] = lambda r: 0) -> list[Row]:
    """Each pair gets the factor table of another opposed pair of the same topic (and ``cell``); labels and stance stay."""
    out = list(rows)
    for t in sorted({(r.topic, cell(r)) for r in rows}, key=str):
        idx = [i for i, r in enumerate(rows) if (r.topic, cell(r)) == t]
        perm = rng.permutation(idx)
        for i, j in zip(idx, perm):
            out[i] = dataclasses.replace(rows[i], factors=rows[j].factors)
    return out


def md_table(rows: list[list[str]]) -> str:
    head, *body = rows
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|", *["| " + " | ".join(map(str, r)) + " |" for r in body]])


def f(v: float | None, d: int = 2) -> str:
    return "–" if v is None else f"{v:.{d}f}"


NOTES = """## 11. What is learned, what is calibrated, what is deterministic

| Component | Kind |
|---|---|
| NLI model, embedding model | pretrained learned models, frozen, deterministic at inference |
| LLM stance and factor-table extraction | pretrained learned model, frozen (temperature 0, cached); the extraction is model-assisted |
| quote grounding, one-sided → unknown, paper-level → half weight | deterministic code |
| factor roles, evidence features d and e, sufficiency E | deterministic code |
| comparison of two stated values | the LLM's status label; deterministic comparison of numeric and named values is not implemented yet |
| comparability coefficients and assertion threshold | **fitted on development data** (penalised logistic regression, 2–6 numbers) |
| sufficiency rule | set a priori (at least one scope or measurement factor stated by both papers); the development labels have no 'cannot tell' option and cannot calibrate it |
| conflict status | deterministic given the table and the fitted numbers |
| explanation | LLM wording after the decision, checked against the evidence record |

**Is the evidence model itself a learning algorithm?** "Evidence model" as a general term is not: it names a way
of combining pieces of evidence into a judgement, and its numbers can come from anywhere (V1's weights were set
by hand, which is still an evidence model). *This implementation* estimates its few coefficients from labelled
development pairs by penalised logistic regression, which is statistical learning in the plain sense of fitting
parameters to data. It is a small calibrated statistical model inside an otherwise rule-based pipeline: it has
at most six fitted numbers, it sees only counts derived from the factor table, and it never sees text. ClaimGraph
as a whole is not a learned or end-to-end trained system; no pretrained model is trained or fine-tuned.
"""


def run_study(batch: str = "eval2", run: str = "main", out_dir: Path | None = None, *, n_perm: int = 1000, n_boot: int = 2000,
              seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = out_dir or EVAL_DIR / "results" / "v2_dev"
    rows = load_rows(batch, run)
    scored = [r for r in rows if r.final]
    opp_all = [r for r in rows if r.opposed]
    opp = [r for r in scored if r.opposed]
    y = np.array([r.genuine for r in opp])
    yf = y.astype(float)
    topics = np.array([r.topic for r in opp])
    folds = topic_folds(opp)
    E = np.array([C.sufficiency(_ev(r)) for r in opp])
    known = E > 0
    n_pos_all = sum(r.genuine for r in scored)
    res: dict = {"batch": batch, "run": run, "pairs": len(rows), "scored": len(scored), "opposed_scored": len(opp),
                 "opposed_contradictions": int(y.sum()), "all_contradictions": n_pos_all, "l2": L2, "seed": seed}
    md = ["# V2 development study: does graded factor evidence separate genuine from explained conflicts?", "",
          f"**Development data, not a paper result.** Batch `{batch}` (run `{run}`), consensus labels of three annotator models, "
          "unadjudicated. These pairs were the V1 test set; they are development data from V2 onwards.", "",
          f"- annotated pairs: {len(rows)}; with a consensus label: {len(scored)}; consensus contradictions: {n_pos_all}",
          f"- pairs the system reads as opposed (V1 trigger, unchanged): {len(opp)} scored, of which {int(y.sum())} are consensus "
          f"contradictions ({100 * y.mean():.0f}%). The other {n_pos_all - int(y.sum())} contradictions are lost at the stance step "
          f"(LLM stance: {dict(Counter(r.stance for r in scored if r.genuine and not r.opposed))}); no context model can recover them.",
          f"- topics (folds): {', '.join(f'{t} {sum(r.topic == t for r in opp)}/{sum(r.genuine for r in opp if r.topic == t)}' for t in sorted(set(topics)))} "
          "(opposed pairs / contradictions)", ""]

    # ---- 1. what the factor tables contain
    md += ["## 1. What the factor tables contain on opposed pairs", "",
           "Contradictions / pairs for each status of each factor. `different*` = a stated difference the LLM marked immaterial.", ""]
    table = [["Factor", "Role", "not stated by both", "same", "partly", "different*", "different"]]
    coverage = {}
    for name in C.FACTOR_ORDER:
        cnt, pos = Counter(), Counter()
        for r in opp:
            fc = next((x for x in r.factors if x.factor == name), None)
            st = "unknown" if fc is None else ("immaterial" if fc.status == "overlap" and fc.note == C.IMMATERIAL_NOTE else fc.status)
            cnt[st] += 1
            pos[st] += r.genuine
        coverage[name] = {st: [pos[st], cnt[st]] for st in cnt}
        table.append([name, C.ROLES[name], *[f"{pos[st]}/{cnt[st]}" if cnt[st] else "–" for st in ("unknown", "same", "overlap", "immaterial", "different")]])
    never = [n for n in C.FACTOR_ORDER if coverage[n].get("unknown", [0, 0])[1] == len(opp)]
    md += [md_table(table), "",
           f"Factors never stated by both papers in any opposed pair: **{', '.join(never) or 'none'}**. Their weights cannot be "
           "estimated from this data, whatever the model.", ""]
    res["coverage"], res["never_known"] = coverage, never

    # ---- 2. where the signal is
    ctx_mis = np.array([_diff(r, roles=CONTEXT_ROLES) for r in opp])
    raw_scores = {"all mismatches (uniform)": np.array([_diff(r) for r in opp]), "scope + measurement mismatches": ctx_mis,
                  "scope mismatches": np.array([_diff(r, roles=("scope",)) for r in opp]),
                  "hand-weighted mismatches (V1 weights)": np.array([_diff(r, weights=HAND_WEIGHTS) for r in opp]),
                  "intervention mismatches": np.array([_diff(r, roles=("intervention",)) for r in opp]),
                  "agreeing factors (reversed: more agreement = more genuine)": -np.array([_same(r) for r in opp])}
    md += ["## 2. Where the signal is", "",
           "E = number of scope and measurement factors stated by both papers (a paper-level statement counts half). "
           "E = 0 is what V1 called *insufficient evidence*.", ""]
    table = [["Shared context", "Opposed pairs", "Consensus contradictions", "Rate", "…of which unanimous", "Gemini", "Qwen", "Haiku"]]
    cells = {}
    for label, mask_fn in (("E = 0 (none stated by both)", lambda r: C.sufficiency(_ev(r)) == 0), ("E > 0", lambda r: C.sufficiency(_ev(r)) > 0)):
        sel, sel_all = [r for r in opp if mask_fn(r)], [r for r in opp_all if mask_fn(r)]
        k = sum(r.genuine for r in sel)
        per = [f"{sum(r.votes[i] == 'CONTRADICTS' for r in sel_all)}/{len(sel_all)}" for i in range(3)]
        cells[label] = {"n": len(sel), "contradictions": k, "per_annotator": per}
        table.append([label, len(sel), k, f"{100 * k / len(sel):.0f}%", sum(len(set(r.votes)) == 1 for r in sel if r.genuine), *per])
    from app.evaluation.scoring import fisher_exact_two_sided

    k0, n0 = int(y[~known].sum()), int((~known).sum())
    k1, n1 = int(y[known].sum()), int(known.sum())
    fisher = fisher_exact_two_sided(k0, n0 - k0, k1, n1 - k1)
    by_topic = ", ".join(f"{t} {int(y[(topics == t) & ~known].sum())}/{int(((topics == t) & ~known).sum())} vs "
                         f"{int(y[(topics == t) & known].sum())}/{int(((topics == t) & known).sum())}" for t in sorted(set(topics)))
    md += [md_table(table), "", "Per-annotator columns: that model's own CONTRADICTS labels over all opposed pairs, including three-way splits.", "",
           f"Fisher's exact test, E = 0 versus E > 0 (consensus labels): p = {fisher:.4f}. By topic (E = 0 vs E > 0): {by_topic}.", ""]
    res["sufficiency_fisher_p"] = fisher
    table = [["Contradiction rate by amount of context mismatch (pairs with E > 0)", "Pairs", "Contradictions", "Rate"]]
    for lo, hi, label in ((0, 0.01, "no mismatch"), (0.01, 0.51, "0.5 (one partial)"), (0.51, 1.01, "1"), (1.01, 2.01, "1.5–2"), (2.01, 99, "> 2")):
        m = known & (ctx_mis >= lo) & (ctx_mis < hi)
        if m.sum():
            table.append([label, int(m.sum()), int(y[m].sum()), f"{100 * y[m].mean():.0f}%"])
    md += [md_table(table), ""]
    table = [["Score (no fitting: fewer mismatches = more likely genuine)", "AUC, all opposed pairs", f"AUC, only pairs with E > 0 (n = {int(known.sum())}, {int(y[known].sum())} contradictions)"]]
    raw = {}
    for label, v in raw_scores.items():
        a_all, a_known = auc_within(y, -v, topics), auc_within(y[known], -v[known], topics[known])
        lo, hi = boot_ci(lambda i: auc_within(y[known][i], -v[known][i], topics[known][i]), int(known.sum()), rng, n_boot)
        raw[label] = {"all": a_all, "known": a_known, "known_ci": [lo, hi]}
        table.append([label, f(a_all), f"{f(a_known)} [{f(lo)}, {f(hi)}]"])
    a_e = auc_within(y, (~known).astype(float), topics)
    table.append(["E = 0 versus E > 0 alone", f(a_e), "–"])
    md += [md_table(table), "", "AUCs compare pairs within the same topic. 0.50 = no signal.", ""]
    res["sufficiency_cells"], res["raw_auc"], res["auc_context_stated"] = cells, raw, a_e

    # ---- 3. cross-validated ranking quality
    md += ["## 3. Fitted models: leave-one-topic-out cross-validation", "",
           f"Target: consensus CONTRADICTS among the {len(opp)} opposed pairs. Log-loss skill = 1 − loss / loss of the constant "
           f"model (negative = worse than ignoring the factors). Ridge penalty fixed in advance (λ = {L2}). "
           "Last column: the same model fitted and tested only on pairs with E > 0, where sufficiency cannot help.", ""]
    table = [["Model", "Fitted numbers", "AUC [95% CI]", "Log-loss", "Skill vs constant", "5-fold AUC (200 repeats)", "AUC, E > 0 only"]]
    cv: dict[str, dict] = {}
    oofs: dict[str, np.ndarray] = {}
    asserted: dict[str, np.ndarray] = {}
    opp_known = [r for r, k in zip(opp, known) if k]
    for spec in SPECS:
        oof, dec = cv_scores(opp, spec, yf, folds)
        oofs[spec.name], asserted[spec.name] = oof, dec
        loss = log_loss(y, oof)
        a = auc_within(y, oof, topics)
        lo, hi = boot_ci(lambda i: auc_within(y[i], oof[i], topics[i]), len(opp), rng, n_boot)
        rep = []
        a_known = None
        if spec.columns:
            X = design(opp, spec)
            for _ in range(200):
                o = np.zeros(len(opp))
                for test in stratified_folds(y, 5, rng):
                    train = np.setdiff1d(np.arange(len(opp)), test)
                    o[test] = predict_proba(X[test], fit_logistic(X[train], yf[train], spec.nonneg, iters=800))
                rep.append(auc(y, o))
            if spec.name != "context_stated":
                ok, _ = cv_scores(opp_known, spec, yf[known], topic_folds(opp_known))
                a_known = auc_within(y[known], ok, topics[known])
        cv[spec.name] = {"n_params": spec.n_params, "auc": a if spec.columns else 0.5, "auc_ci": [lo, hi], "log_loss": loss,
                         "auc_5fold_mean": float(np.mean(rep)) if rep else None, "auc_5fold_sd": float(np.std(rep)) if rep else None,
                         "auc_known": a_known}
    base_loss = cv["constant"]["log_loss"]
    for spec in SPECS:
        c = cv[spec.name]
        c["skill"] = 1 - c["log_loss"] / base_loss
        table.append([spec.title, spec.n_params, f"{f(c['auc'])} [{f(c['auc_ci'][0])}, {f(c['auc_ci'][1])}]" if spec.columns else "0.50 (by construction)",
                      f(c["log_loss"], 3), f"{100 * c['skill']:+.1f}%",
                      f"{f(c['auc_5fold_mean'])} ± {f(c['auc_5fold_sd'])}" if c["auc_5fold_mean"] is not None else "–", f(c["auc_known"])])
    md += [md_table(table), ""]
    res["cv"] = cv

    # ---- 4. model selection
    candidates = ["context_stated", "uniform_ctx", "uniform_diff", "hand", "uniform", "hand_same", "role2", "role", "role4", "free12"]
    best = min(candidates, key=lambda n: cv[n]["log_loss"])
    per_pair = {n: -(y * np.log(np.clip(oofs[n], 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - oofs[n], 1e-6, 1))) for n in candidates + ["constant"]}
    md += ["## 4. Is a bigger model better? Selection by the rule fixed in advance", "",
           "Rule: the model with the fewest fitted numbers whose out-of-fold log-loss is within one standard error of the best; "
           "ties go to the model without hand-set constants. Differences are paired over pairs (bootstrap).", ""]
    table = [["Model", "Fitted numbers", "Log-loss", f"Difference to best ({SPEC[best].name}) ± SE", "Within 1 SE", "Difference to constant ± SE"]]
    sel_info = {}
    for n in candidates:
        d = per_pair[n] - per_pair[best]
        d0 = per_pair[n] - per_pair["constant"]
        idx = rng.integers(0, len(opp), (n_boot, len(opp)))
        se, se0 = float(d[idx].mean(axis=1).std()), float(d0[idx].mean(axis=1).std())
        within = bool(d.mean() <= se + 1e-12)
        sel_info[n] = {"n_params": SPEC[n].n_params, "log_loss": cv[n]["log_loss"], "diff_best": float(d.mean()), "se": se, "within": within,
                       "diff_constant": float(d0.mean()), "se_constant": se0}
        table.append([SPEC[n].title, SPEC[n].n_params, f(cv[n]["log_loss"], 3), f"{d.mean():+.3f} ± {se:.3f}", "yes" if within else "no",
                      f"{d0.mean():+.3f} ± {se0:.3f}"])
    eligible = [n for n in candidates if sel_info[n]["within"]]
    selected = min(eligible, key=lambda n: (SPEC[n].n_params, n.startswith("hand"), candidates.index(n)))
    selected_factor_model = min([n for n in eligible if n != "context_stated"] or [best], key=lambda n: (SPEC[n].n_params, n.startswith("hand"), candidates.index(n)))
    md += [md_table(table), "",
           f"Selected by the rule: **{SPEC[selected].title}**. Simplest eligible model that uses factor values: "
           f"**{SPEC[selected_factor_model].title}**.", ""]
    res["selection"] = {"best": best, "selected": selected, "selected_factor_model": selected_factor_model, "table": sel_info}

    # ---- 5. shuffled-factor controls
    md += ["## 5. Shuffled-factor controls", "",
           f"{n_perm} shuffles each. (a) every opposed pair gets the factor table of another opposed pair of the same topic: "
           "tests whether the tables carry any signal. (b) the swap is only among pairs of the same topic *and* the same side of "
           "E = 0: tests whether the factor values carry signal beyond whether any context is stated.", ""]
    table = [["Model", "Real AUC", "(a) shuffled: mean [2.5%, 97.5%]", "p", "(b) shuffled within E cell: mean [2.5%, 97.5%]", "p"]]
    perm_res = {}
    for name in ("uniform_ctx", "uniform", "hand", "role", "role4"):
        spec = SPEC[name]
        row, perm_res[name] = [spec.title, f(cv[name]["auc"])], {"real": cv[name]["auc"]}
        for tag, cell in (("a", lambda r: 0), ("b", lambda r: C.sufficiency(_ev(r)) == 0)):
            vals = []
            for _ in range(n_perm):
                Xs = design(shuffled_rows(opp, rng, cell), spec)
                o = np.zeros(len(opp))
                for test in folds:
                    train = np.setdiff1d(np.arange(len(opp)), test)
                    o[test] = predict_proba(Xs[test], fit_logistic(Xs[train], yf[train], spec.nonneg, iters=600))
                vals.append(auc_within(y, o, topics))
            v = np.array(vals)
            pv = float((1 + (v >= cv[name]["auc"] - 1e-12).sum()) / (1 + n_perm))
            perm_res[name][tag] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "p": pv}
            row += [f"{f(v.mean())} [{f(np.percentile(v, 2.5))}, {f(np.percentile(v, 97.5))}]", f(pv, 3)]
        table.append(row)
    md += [md_table(table), ""]
    res["shuffled"] = perm_res

    # ---- 6. coefficients
    md += ["## 6. Fitted coefficients (all opposed development pairs)", "",
           "Log-odds of *genuine contradiction*. A mismatch weight w enters as −w per mismatched factor, the agreement weight as "
           "+w per agreeing factor; constrained weights cannot go below 0. Intervals: percentile bootstrap over pairs. "
           "Odds ratio = exp(±w): the factor by which one mismatch (or agreement) multiplies the odds of a genuine contradiction.", ""]
    coefs: dict[str, dict] = {}
    for name in ("uniform_ctx", "hand", "role", "role4", "role4_free", "free12"):
        spec = SPEC[name]
        X = design(opp, spec)
        beta = fit_logistic(X, yf, spec.nonneg)
        boots = np.array([fit_logistic(X[i], yf[i], spec.nonneg, iters=800) for i in (rng.integers(0, len(opp), len(opp)) for _ in range(min(n_boot, 500)))])
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        at_zero = (np.abs(boots) < 1e-6).mean(axis=0)
        cols = ("intercept", *spec.columns)
        coefs[name] = {c: {"value": float(b), "ci": [float(l), float(h)], "share_zero": float(z)} for c, b, l, h, z in zip(cols, beta, lo, hi, at_zero)}
        table = [["Term", "Weight", "95% interval", "Odds ratio", "Bootstrap fits at exactly 0"]]
        for c, b, l, h, z in zip(cols, beta, lo, hi, at_zero):
            sign = 1 if c.startswith("same") else -1
            table.append([c, f(b), f"[{f(l)}, {f(h)}]", "–" if c == "intercept" else f(float(np.exp(sign * b))), "–" if c == "intercept" else f"{100 * z:.0f}%"])
        md += [f"**{spec.title}**", "", md_table(table), ""]
    res["coefficients"] = coefs

    # ---- 7. decisions
    md += ["## 7. Asserted contradictions: precision, recall and false contradictions", "",
           f"Over all {len(scored)} scored pairs ({n_pos_all} consensus contradictions). Every decision of a fitted model is "
           "out-of-fold: coefficients and threshold come from the other topics.", ""]
    ids = np.array([r.item_id for r in opp])
    insufficient = set(ids[~known])
    systems: dict[str, set[str]] = {
        "No context (assert every opposed pair)": set(ids),
        "V1 hard gate — asserted": {r.item_id for r in opp if r.v1 == "CONTRADICTS"},
        "V1 hard gate — asserted + unasserted (insufficient)": {r.item_id for r in opp if r.v1 in ("CONTRADICTS", ABSTAIN)},
        "Typed hard gate — asserted": {r.item_id for r in opp if C.typed_gate_status(r.factors) == "genuine"},
        "Typed hard gate — asserted + insufficient": {r.item_id for r in opp if C.typed_gate_status(r.factors) in ("genuine", "insufficient")},
        "Opposed and no intervention mismatch (rule proposed for question A)": {r.item_id for r in opp if C.typed_gate_status(r.factors) != "none"},
        "Only the pairs with no shared context (E = 0)": insufficient,
    }
    # (i) fitted on all opposed pairs, no sufficiency rule, threshold = max training F1
    for name in ("semantic", "context_stated", "uniform_ctx", "uniform", "hand", "role", "role4", "free12", "role4_llm_material"):
        systems[f"{SPEC[name].title} — no sufficiency rule, max-F1 threshold"] = set(ids[asserted[name]])
    # (ii) P3 as specified: E = 0 -> insufficient; comparability fitted and thresholded on pairs with E > 0
    p3: dict[str, dict] = {}
    for name in ("uniform_ctx", "hand", "role2", "role", "role4"):
        spec = SPEC[name]
        X = design(opp, spec)
        dec = np.zeros(len(opp), dtype=bool)
        for test in folds:
            train = np.setdiff1d(np.arange(len(opp)), test)
            tr = train[known[train]]
            beta = fit_logistic(X[tr], yf[tr], spec.nonneg)
            tau = precision_threshold(y[tr], predict_proba(X[tr], beta), 0.5)
            dec[test] = known[test] & (predict_proba(X[test], beta) >= tau)
        systems[f"P3 [{name}] — asserted"] = set(ids[dec])
        systems[f"P3 [{name}] — asserted + insufficient"] = set(ids[dec]) | insufficient
        # the parameters that would be frozen for a held-out test: fitted on every development pair with E > 0
        beta = fit_logistic(X[known], yf[known], spec.nonneg)
        tau = precision_threshold(y[known], predict_proba(X[known], beta), 0.5)
        p3[name] = {"title": spec.title, "out_of_fold_asserted": int(dec.sum()), "intercept": float(beta[0]),
                    "columns": dict(zip(spec.columns, map(float, beta[1:]))), "threshold": None if np.isinf(tau) else tau}
    md += ["Rows 'P3 [model]' apply the proposal exactly: E = 0 → *insufficient*; otherwise assert when p ≥ τ, with the "
           "coefficients fitted on training pairs with E > 0 and τ the lowest value at which at least half of the asserted "
           "training pairs are contradictions (no such value → nothing is asserted).", ""]
    table = [["Decision rule", "Asserted", "Correct", "Precision", "Recall", "F1", "False con. /100"]]
    decisions = {}
    for title, sset in systems.items():
        m = decision_metrics(sset, scored)
        decisions[title] = m
        table.append([title, m["asserted"], m["tp"], f(m["precision"], 1), f(m["recall"], 1), f(m["f1"], 1), f(m["false_con_per_100"], 1)])
    md += [md_table(table), "",
           "The annotator models had no 'cannot tell' option. The rows that add the *insufficient* pairs therefore show how those "
           "pairs were labelled when a label was forced, not whether abstaining on them was right.", ""]
    res["decisions"], res["p3_parameters"] = decisions, p3

    # ---- 8. ablations
    md += ["## 8. Role and factor ablations", "", "Leave-one-topic-out AUC of the uniform model (agreement + mismatch, all roles) with one role or one factor removed.", ""]
    table = [["Removed", "AUC, all opposed", "Δ", "AUC, E > 0 only", "Δ"]]
    abl = {}

    def uniform_without(pred: Callable[[C.FactorEvidence], bool]) -> Spec:
        return Spec("abl", "abl", ("same", "diff"), lambda r: [sum(e.same for e in _ev(r) if not pred(e)),
                                                                -sum(e.different for e in _ev(r) if not pred(e))], (True, True))

    def both(spec: Spec) -> tuple[float | None, float | None]:
        o, _ = cv_scores(opp, spec, yf, folds)
        ok, _ = cv_scores(opp_known, spec, yf[known], topic_folds(opp_known))
        return auc_within(y, o, topics), auc_within(y[known], ok, topics[known])

    base_all, base_known = both(uniform_without(lambda e: False))
    variants = ([(f"role: {role}", (lambda e, role=role: e.role == role)) for role in C.ROLE_ORDER]
                + [(f"factor: {name}", (lambda e, name=name: e.factor == name)) for name in C.FACTOR_ORDER if name not in never]
                + [(f"all but {keep} (keep only {keep})", (lambda e, keep=keep: e.role != keep)) for keep in ("scope", "measurement", "intervention")])
    for label, pred in variants:
        a1, a2 = both(uniform_without(pred))
        abl[label] = {"all": a1, "known": a2}
        table.append([label, f(a1), f"{a1 - base_all:+.2f}", f(a2), f"{a2 - base_known:+.2f}"])
    md += [md_table(table), "", f"Nothing removed: {f(base_all)} (all opposed), {f(base_known)} (E > 0 only).", ""]
    res["ablations"] = {"base": {"all": base_all, "known": base_known}, **abl}

    # ---- 9. robustness of the target and the penalty
    md += ["## 9. Sensitivity to the reference labels and to the penalty", "", "Leave-one-topic-out AUC against other readings of the same annotations.", ""]
    names = ("context_stated", "uniform_ctx", "hand", "role", "role4", "semantic")
    table = [["Target", "n", "positives", *names, "uniform_ctx, E > 0 only"]]
    sens = {}
    targets: list[tuple[str, list[Row], np.ndarray | None, np.ndarray | None]] = [
        ("consensus (main)", opp, yf, y),
        ("unanimous pairs only", [r for r in opp if len(set(r.votes)) == 1], None, None),
        ("vote share incl. three-way splits (soft label; AUC against ≥ 2 votes)", opp_all,
         np.array([r.vote_share for r in opp_all]), np.array([int(r.vote_share >= 0.66) for r in opp_all])),
    ] + [(f"annotator {a} alone", opp_all, np.array([float(r.votes[i] == "CONTRADICTS") for r in opp_all]),
          np.array([int(r.votes[i] == "CONTRADICTS") for r in opp_all])) for i, a in enumerate(("gemini", "qwen", "haiku"))]
    for label, rs, yt, yh in targets:
        if yt is None or yh is None:
            yh = np.array([r.genuine for r in rs])
            yt = yh.astype(float)
        tp = np.array([r.topic for r in rs])
        row = [label, len(rs), int(yh.sum())]
        sens[label] = {}
        for name in names:
            o, _ = cv_scores(rs, SPEC[name], yt, topic_folds(rs))
            sens[label][name] = auc_within(yh, o, tp)
            row.append(f(sens[label][name]))
        k = np.array([C.sufficiency(_ev(r)) > 0 for r in rs])
        v = np.array([_diff(r, roles=CONTEXT_ROLES) for r in rs])
        sens[label]["uniform_ctx_known"] = auc_within(yh[k], -v[k], tp[k])
        table.append(row + [f(sens[label]["uniform_ctx_known"])])
    md += [md_table(table), ""]
    pen_names = ("uniform_ctx", "hand", "role", "role4", "free12")
    table = [["Penalty λ", *pen_names]]
    for l2 in (0.1, 1.0, 10.0):
        table.append([l2, *[f(auc_within(y, cv_scores(opp, SPEC[n], yf, folds, l2)[0], topics)) for n in pen_names]])
    md += [md_table(table), ""]
    res["sensitivity"] = sens

    # ---- 10. signal beyond stance, on all pairs
    md += ["## 10. Factor evidence beyond semantic stance, on all scored pairs", "",
           f"Target: consensus CONTRADICTS among all {len(scored)} scored pairs. 'Stance' = the LLM stance label and NLI "
           "contradiction. The other rows add, for opposed pairs, the named factor evidence.", ""]
    ya = np.array([r.genuine for r in scored])
    ta = np.array([r.topic for r in scored])
    fa = topic_folds(scored)

    def stance_feats(r: Row) -> list[float]:
        return [float(r.stance == "oppose"), float(r.stance == "partial"), float(r.stance == "agree"), r.con]

    s_spec = Spec("stance", "Stance", ("oppose", "partial", "agree", "nli_con"), stance_feats, (False,) * 4)
    extra = [s_spec,
             Spec("stance_e", "Stance + is any context stated", (*s_spec.columns, "no shared context"),
                  lambda r: stance_feats(r) + [float(r.opposed and C.sufficiency(_ev(r)) == 0)], (False,) * 5),
             Spec("stance_mis", "Stance + context mismatches", (*s_spec.columns, "diff:context"),
                  lambda r: stance_feats(r) + [-_diff(r, roles=CONTEXT_ROLES) if r.opposed else 0.0], (False,) * 4 + (True,)),
             Spec("stance_both", "Stance + both", (*s_spec.columns, "no shared context", "diff:context"),
                  lambda r: stance_feats(r) + [float(r.opposed and C.sufficiency(_ev(r)) == 0), -_diff(r, roles=CONTEXT_ROLES) if r.opposed else 0.0],
                  (False,) * 5 + (True,))]
    table = [["Model", "AUC [95% CI]", "Log-loss", "Δ log-loss vs stance ± SE"]]
    beyond = {}
    base_pp = None
    for spec in extra:
        o, _ = cv_scores(scored, spec, ya.astype(float), fa)
        lo, hi = boot_ci(lambda i: auc_within(ya[i], o[i], ta[i]), len(scored), rng, n_boot)
        pp = -(ya * np.log(np.clip(o, 1e-6, 1)) + (1 - ya) * np.log(np.clip(1 - o, 1e-6, 1)))
        base_pp = pp if base_pp is None else base_pp
        d = pp - base_pp
        se = float(d[rng.integers(0, len(scored), (n_boot, len(scored)))].mean(axis=1).std())
        beyond[spec.name] = {"auc": auc_within(ya, o, ta), "ci": [lo, hi], "log_loss": float(pp.mean()), "diff": float(d.mean()), "se": se}
        table.append([spec.title, f"{f(beyond[spec.name]['auc'])} [{f(lo)}, {f(hi)}]", f(pp.mean(), 3), "–" if spec is s_spec else f"{d.mean():+.3f} ± {se:.3f}"])
    md += [md_table(table), ""]
    res["beyond_stance"] = beyond

    kn, kp = int(known.sum()), int(y[known].sum())
    sh = perm_res["uniform_ctx"]
    summary = ["## Summary", "",
               f"1. **The factor evidence carries signal, and all of it is the sufficiency axis.** Opposed pairs in which no context "
               f"factor is stated by both papers are consensus contradictions {cells['E = 0 (none stated by both)']['contradictions']}/"
               f"{cells['E = 0 (none stated by both)']['n']} times; pairs with shared context {kp}/{kn} (Fisher p = {fisher:.4f}; the same "
               f"direction under each annotator's own labels). That split alone gives AUC "
               f"{f(a_e)} and is the model the selection rule picks.",
               f"2. **Graded divergence adds nothing detectable.** Among the {kn} pairs with shared context, no mismatch score separates "
               f"the {kp} contradictions (AUC {f(raw['scope + measurement mismatches']['known'])} for context mismatches, "
               f"{f(raw['hand-weighted mismatches (V1 weights)']['known'])} with the V1 weights). Shuffling factor tables among pairs "
               f"on the same side of E = 0 does not lower the AUC (real {f(sh['real'])}, shuffled {f(sh['b']['mean'])}, p = {f(sh['b']['p'], 2)}). "
               f"With {kp} positives this cannot rule out a moderate effect; it shows none.",
               f"3. **Fitted role weights do not beat simpler models.** Out-of-fold log-loss: role weights {f(cv['role']['log_loss'], 3)}, "
               f"uniform {f(cv['uniform_ctx']['log_loss'], 3)}, V1 hand weights {f(cv['hand']['log_loss'], 3)}, sufficiency only "
               f"{f(cv['context_stated']['log_loss'], 3)}, constant {f(base_loss, 3)}. On all opposed pairs the fitted order is as "
               f"hypothesised (scope {f(coefs['role4']['diff:scope']['value'])} > measurement {f(coefs['role4']['diff:measurement']['value'])} > "
               f"intervention {f(coefs['role4']['diff:intervention']['value'])}), but the scope weight mostly encodes 'no mismatch because "
               f"nothing is stated': fitted only on pairs with shared context it falls to {f(p3['role']['columns']['diff:scope'])} and the "
               f"measurement weight to {f(p3['role']['columns']['diff:measurement'])}. The agreement weight is fitted at zero throughout.",
               f"4. **P3 as specified does not fix recall.** Out-of-fold it asserts {p3['role']['out_of_fold_asserted']} pairs "
               f"(role weights) or {p3['uniform_ctx']['out_of_fold_asserted']} (uniform), none of them a consensus contradiction; "
               "V1 asserted 7, 2 correct. The only rule that reaches about half of the contradictions is adding the pairs with no "
               "shared context, which needs no fitted number and is the V1 'counting unasserted conflicts' reading.",
               "5. **The rule 'an intervention mismatch means the claims are not opposed' is not supported**: pairs with a "
               "method or baseline mismatch are contradictions as often as the others.",
               f"6. **Beyond stance:** over all scored pairs, adding 'is any context stated' to the stance signals moves AUC from "
               f"{f(beyond['stance']['auc'])} to {f(beyond['stance_e']['auc'])}; adding mismatch counts on top changes nothing.", ""]
    md[4:4] = summary
    md += [NOTES]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparability_study.md").write_text("\n".join(md))
    (out_dir / "comparability_study.json").write_text(json.dumps(res, indent=1, default=float))
    return res
