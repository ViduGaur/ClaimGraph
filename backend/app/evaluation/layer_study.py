"""Layer-by-layer study on the C4 four-state labels (experiments E1-E3).

E1  opposition detection   independent readings of the pair (pair-analysis stance, cached five-way relation call, NLI):
                           each alone, agreement rules, and a calibrated combination (leave-one-topic-out)
E2  factor layer           does the twelve-factor table name the factors the annotators name, does it add signal beyond
                           the semantic layer, and does any definition of evidence sufficiency track theirs
E3  conditional prediction per-claim frames (``app.reasoning.frames``) compared in code

Reference: ``eval/annotation/c4/reference.json`` (unadjudicated model consensus); pairs without a consensus status are
left out of every status score. All eight topics are development data, so nothing here is a held-out result. No LLM is
called: E3 reads the frames stored by ``frame_extraction``.

Intervals are 95% bootstrap intervals that resample paper pairs. Fitted scores are out-of-fold, leaving one topic out.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.config import Settings
from app.evaluation import comparability_study as S
from app.evaluation import fourstate_study as FS
from app.evaluation import frame_extraction as FX
from app.evaluation.experiment import load_topic
from app.providers.models import get_models
from app.reasoning import comparability as C
from app.reasoning import frames as FR

f, md_table, pct = S.f, S.md_table, FS.pct
UP = FS.UP
ANNOTATORS = FS.ANNOTATOR_IDS
STRATUM_WEIGHT = {"no_signal": 97 / 30}  # the only stratum that was subsampled (30 of 97 pairs)


@dataclass
class Pair:
    row: FS.Row
    rec: dict
    cluster: str  # the two papers
    a: object
    b: object

    def __getattr__(self, name):  # the four-state row's fields
        return getattr(self.row, name)


def load_pairs(batch: str = "c4") -> list[Pair]:
    rows = FS.load_rows(batch)
    datas: dict = {}
    out = []
    for r in rows:
        key = (r.ref["run"], r.topic)
        if key not in datas:
            d = load_topic(*key)
            datas[key] = (d, {x["pair_id"]: x for x in d.records})
        d, recs = datas[key]
        rec = recs[r.pair_id]
        a, b = d.claims[rec["a"]], d.claims[rec["b"]]
        out.append(Pair(r, rec, "|".join(sorted((a.paper_id, b.paper_id))), a, b))
    return out


# ------------------------------------------------------------------------------------------------ statistics
class Boot:
    """Bootstrap that resamples paper pairs."""

    def __init__(self, clusters: list[str], n_boot: int, seed: int):
        ids = sorted(set(clusters))
        members = {c: np.flatnonzero(np.array(clusters) == c) for c in ids}
        rng = np.random.default_rng(seed)
        self.samples = [np.concatenate([members[ids[j]] for j in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot)]

    def ci(self, fn: Callable[[np.ndarray], float | None]) -> tuple[float | None, float | None]:
        vals = [v for v in (fn(i) for i in self.samples) if v is not None and not np.isnan(v)]
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (None, None)


def prf(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict[str, float | None]:
    tp, n_pred, n_pos = float((w * (y & pred)).sum()), float((w * pred).sum()), float((w * y).sum())
    return {"precision": tp / n_pred if n_pred else None, "recall": tp / n_pos if n_pos else None,
            "f1": 2 * tp / (n_pred + n_pos) if n_pred + n_pos else None}


def binary(y: np.ndarray, pred: np.ndarray, w: np.ndarray, boot: Boot) -> dict:
    y, pred = y.astype(bool), pred.astype(bool)
    out: dict = {"flagged": int(pred.sum()), "correct": int((y & pred).sum()), "positives": int(y.sum())}
    base = prf(y, pred, w)
    for k in ("precision", "recall", "f1"):
        out[k] = {"value": base[k], "ci": boot.ci(lambda i, k=k: prf(y[i], pred[i], w[i])[k])}
    return out


def cell(m: dict) -> str:
    return "–" if m["value"] is None else pct(m["value"], *m["ci"])


def auc_ci(y: np.ndarray, s: np.ndarray, boot: Boot) -> tuple[float | None, tuple]:
    y, s = y.astype(float), s.astype(float)
    return S.auc(y, s), boot.ci(lambda i: S.auc(y[i], s[i]))


def auc_cell(y: np.ndarray, s: np.ndarray, boot: Boot) -> str:
    a, (lo, hi) = auc_ci(y, s, boot)
    return "–" if a is None else f"{a:.2f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")


def loto(X: np.ndarray, y: np.ndarray, topics: np.ndarray, l2: float = S.L2) -> np.ndarray:
    """Out-of-fold probabilities of a ridge-penalised logistic model, leaving one topic out. Signs are free."""
    X1 = np.c_[np.ones(len(X)), X]
    oof = np.zeros(len(y))
    for t in np.unique(topics):
        test = topics == t
        beta = S.fit_logistic(X1[~test], y[~test].astype(float), tuple(False for _ in range(X.shape[1])), l2)
        oof[test] = S.predict_proba(X1[test], beta)
    return oof


def full_fit(X: np.ndarray, y: np.ndarray, l2: float = S.L2) -> list[float]:
    return [round(float(v), 3) for v in S.fit_logistic(np.c_[np.ones(len(X)), X], y.astype(float), tuple(False for _ in range(X.shape[1])), l2)]


def four_state(gold: list[str], pred: list[str], boot_idx: Boot, w: np.ndarray | None = None) -> dict:
    g, p = np.array(gold), np.array(pred)
    res = FS.evaluate4(gold, pred)
    res["accuracy_weighted"] = float((w * (g == p)).sum() / w.sum()) if w is not None else None
    res["accuracy_ci"] = boot_idx.ci(lambda i: float((g[i] == p[i]).mean()))
    res["macro_f1_ci"] = boot_idx.ci(lambda i: FS.evaluate4(list(g[i]), list(p[i]))["macro_f1"])
    return res


def four_row(label: str, res: dict) -> list[str]:
    return [label, pct(res["accuracy"], *res["accuracy_ci"]), pct(res.get("accuracy_weighted")), pct(res["macro_f1"], *res["macro_f1_ci"]),
            f"{res['GENUINE']['pred']} / {res['GENUINE']['tp']}", f"{res['CONTEXT_DEPENDENT']['pred']} / {res['CONTEXT_DEPENDENT']['tp']}",
            f"{res['INSUFFICIENT_EVIDENCE']['pred']} / {res['INSUFFICIENT_EVIDENCE']['tp']}", f"{res['NONE']['pred']} / {res['NONE']['tp']}"]


FOUR_HEAD = ["System", "Accuracy", "Accuracy, stratum-weighted", "Macro-F1", "GENUINE said / right", "CONTEXT_DEP. said / right", "INSUFFICIENT said / right", "NONE said / right"]


# ======================================================================================================= E1
def e1_signals(pairs: list[Pair]) -> dict[str, np.ndarray]:
    return {
        "trigger": np.array([p.opposed for p in pairs]),
        "stance_oppose": np.array([p.stance == "oppose" for p in pairs]),
        "stance_partial": np.array([p.stance == "partial" for p in pairs]),
        "rel_ctx": np.array([p.direct["context"] == "CONTRADICTS" for p in pairs]),
        "rel_claims": np.array([p.direct["claims"] == "CONTRADICTS" for p in pairs]),
        "nli_label": np.array([p.nli_label == "CONTRADICTS" for p in pairs]),
        "nli_con": np.array([p.con for p in pairs]),
        "nli_con_max": np.array([max(p.rec["nli_ab"]["contradiction"], p.rec["nli_ba"]["contradiction"]) for p in pairs]),
        "nli_ent": np.array([p.ent for p in pairs]),
        "sim": np.array([p.sim for p in pairs]),
    }


E1_MODELS: dict[str, tuple[str, list[str]]] = {
    "cal_stance": ("Calibrated: pair-analysis stance only", ["stance_oppose", "stance_partial"]),
    "cal_nli": ("Calibrated: NLI scores only (continuous)", ["nli_con", "nli_ent"]),
    "cal_stance_nli": ("Calibrated: stance + NLI scores", ["stance_oppose", "stance_partial", "nli_con", "nli_ent"]),
    "cal_two": ("Calibrated: stance + relation call", ["stance_oppose", "rel_ctx"]),
    "cal_three": ("Calibrated: stance + both relation calls", ["stance_oppose", "rel_ctx", "rel_claims"]),
    "cal_three_nli": ("Calibrated: stance + both relation calls + NLI scores", ["stance_oppose", "rel_ctx", "rel_claims", "nli_con", "nli_ent"]),
}


def decision_layers() -> dict[str, tuple[str, Callable[[Pair], str]]]:
    return {"cd": ("always CONTEXT_DEPENDENT", lambda p: "context_dependent"), "v1": ("V1 hard gate", lambda p: C.v1_status(p.factors))}


def run_e1(pairs: list[Pair], boot: Boot, sboot: Boot, scored: np.ndarray) -> tuple[dict, list[str], dict[str, np.ndarray]]:
    y = np.array([p.ref["stance"] == "oppose" for p in pairs])
    w = np.array([STRATUM_WEIGHT.get(p.stratum, 1.0) for p in pairs])
    topics = np.array([p.topic for p in pairs])
    sig = e1_signals(pairs)
    votes = sig["stance_oppose"].astype(int) + sig["rel_ctx"] + sig["rel_claims"]
    detectors: dict[str, tuple[str, np.ndarray]] = {
        "current": ("Current trigger (stance oppose, or partial + NLI)", sig["trigger"]),
        "stance": ("Pair-analysis stance = oppose", sig["stance_oppose"]),
        "rel_ctx": ("Relation call with passages = CONTRADICTS", sig["rel_ctx"]),
        "rel_claims": ("Relation call, claims only = CONTRADICTS", sig["rel_claims"]),
        "nli": ("NLI only", sig["nli_label"]),
        "agree": ("Both agree: stance oppose AND relation call", sig["stance_oppose"] & sig["rel_ctx"]),
        "either": ("Either: stance oppose OR relation call", sig["stance_oppose"] | sig["rel_ctx"]),
        "vote2": ("At least 2 of 3 readings (stance, two relation calls)", votes >= 2),
        "vote3": ("All 3 readings", votes >= 3),
    }
    res: dict = {"n": len(pairs), "opposed_by_reference": int(y.sum()), "detectors": {}, "calibrated": {}}
    lines = ["## E1. Opposition detection", "",
             f"Target: the annotators' majority reading of face-value opposition (dimension A), all {len(pairs)} pairs, "
             f"{int(y.sum())} opposed. Precision, recall and F1 weight the subsampled stratum by 97/30. "
             "\"Flagged / right\" are raw counts.", ""]
    table = [["Detector", "Flagged / right", "Precision", "Recall", "F1", "Topics where precision ≥ current", "GENUINE pairs kept (of 4)"]]
    genuine = np.array([p.status == "GENUINE" for p in pairs])
    cur_prec = {t: prf(y[topics == t], sig["trigger"][topics == t], w[topics == t])["precision"] for t in np.unique(topics)}
    oof: dict[str, np.ndarray] = {}

    def add(name: str, label: str, pred: np.ndarray) -> None:
        m = binary(y, pred, w, boot)
        better = sum(1 for t in np.unique(topics)
                     if (prf(y[topics == t], pred[topics == t], w[topics == t])["precision"] or 0) >= (cur_prec[t] or 0) and pred[topics == t].any())
        m["topics_not_worse"] = better
        m["genuine_kept"] = int((pred.astype(bool) & genuine).sum())
        m["flagged_in_subsampled_stratum"] = int((pred.astype(bool) & (w > 1)).sum())
        res["detectors"][name] = m
        table.append([label, f"{m['flagged']} / {m['correct']}", cell(m["precision"]), cell(m["recall"]), cell(m["f1"]), f"{better} of {len(cur_prec)}", str(m["genuine_kept"])])

    for name, (label, pred) in detectors.items():
        add(name, label, pred)
    cal_rows = [["Calibrated score (leave-one-topic-out)", "AUC pooled", "AUC within topic", "Log-loss", "Coefficients on all pairs (intercept first)"]]
    for name, (label, cols) in E1_MODELS.items():
        X = np.c_[tuple(sig[c].astype(float) for c in cols)]
        p = loto(X, y, topics)
        oof[name] = p
        add(name, label + ", p ≥ 0.5", p >= 0.5)
        a, (lo, hi) = auc_ci(y, p, boot)
        aw = S.auc_within(y.astype(float), p, topics)
        res["calibrated"][name] = {"features": cols, "auc": a, "auc_ci": [lo, hi], "auc_within": aw, "log_loss": S.log_loss(y.astype(float), p), "coefficients": full_fit(X, y)}
        cal_rows.append([label, auc_cell(y, p, boot), f(aw), f(S.log_loss(y.astype(float), p), 3), ", ".join(f"{v:+.2f}" for v in full_fit(X, y))])
    lines += [md_table(table), "", md_table(cal_rows), "",
              f"Constant log-loss {S.log_loss(y.astype(float), np.full(len(y), y.mean())):.3f}. "
              "A calibrated score is asked to do one thing here: beat the rules above it out of fold.", ""]

    # paired differences, same bootstrap samples
    def diff(a: np.ndarray, b: np.ndarray, k: str) -> str:
        base = (prf(y, a, w)[k] or 0) - (prf(y, b, w)[k] or 0)
        lo, hi = boot.ci(lambda i: (prf(y[i], a[i], w[i])[k] or 0) - (prf(y[i], b[i], w[i])[k] or 0))
        return f"{100 * base:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"

    cmp_rows = [["Compared", "Δ precision", "Δ recall", "Δ F1"]]
    best_cal = oof["cal_three"] >= 0.5
    for label, a, b in [("Both agree − current", detectors["agree"][1], sig["trigger"]),
                        ("Relation call − current", sig["rel_ctx"], sig["trigger"]),
                        ("Both agree − relation call", detectors["agree"][1], sig["rel_ctx"]),
                        ("2 of 3 − relation call", detectors["vote2"][1], sig["rel_ctx"]),
                        ("2 of 3 − current", detectors["vote2"][1], sig["trigger"]),
                        ("Calibrated (stance + both calls) − current", best_cal, sig["trigger"]),
                        ("Calibrated (stance + both calls) − relation call", best_cal, sig["rel_ctx"]),
                        ("Calibrated (stance + both calls) − both agree", best_cal, detectors["agree"][1]),
                        ("Calibrated with NLI − without", oof["cal_three_nli"] >= 0.5, best_cal)]:
        cmp_rows.append([label, diff(a, b, "precision"), diff(a, b, "recall"), diff(a, b, "f1")])
        res.setdefault("differences", {})[label] = cmp_rows[-1][1:]
    lines += ["Paired differences (percentage points, same bootstrap samples):", "", md_table(cmp_rows), ""]

    # where the false and the missed opposition come from
    so = sig["trigger"]
    origin = Counter(("stance oppose" if p.stance == "oppose" else f"{p.stance} + NLI") for p in pairs if p.opposed)
    wrong = Counter(("stance oppose" if p.stance == "oppose" else f"{p.stance} + NLI") for p, yy in zip(pairs, y) if p.opposed and not yy)
    missed = Counter(str(p.stance) for p, yy in zip(pairs, y) if not p.opposed and yy)
    una = Counter(p.ref["stance_agreement"] for p, yy in zip(pairs, y) if p.opposed and not yy)
    res["origin"] = {"system_opposed": dict(origin), "not_confirmed": dict(wrong), "not_confirmed_agreement": dict(una), "missed_by_system_stance": dict(missed)}
    lines += [f"Where the {int(so.sum())} → {int((so & y).sum())} gap comes from: " + "; ".join(f"{k}: {origin[k]} pairs, {wrong[k]} not confirmed" for k in origin)
              + f". Of the {int((so & ~y).sum())} unconfirmed pairs, {una.get('unanimous', 0)} are unanimous \"not opposed\". "
              + f"The system misses {int((~so & y).sum())} opposed pairs; its stance for them: {dict(missed)}.", ""]
    sig_rows = [["Signal", f"AUC, all {len(pairs)} pairs", f"AUC, {int(so.sum())} system-opposed pairs"]]
    sub = Boot([p.cluster for p in pairs if p.opposed], len(boot.samples), 5)
    for name in ("rel_ctx", "rel_claims", "nli_con", "nli_con_max", "nli_ent", "sim"):
        sig_rows.append([name, auc_cell(y, sig[name], boot), auc_cell(y[so], sig[name][so], sub)])
    lines += [md_table(sig_rows), ""]

    # end to end: each detector in front of the two simplest decision layers
    gold = [p.status for p in pairs if p.status]
    sp = [p for p in pairs if p.status]
    sw = np.array([STRATUM_WEIGHT.get(p.stratum, 1.0) for p in sp])
    e2e: dict = {}
    table = [FOUR_HEAD, four_row("Constant NONE", four_state(gold, ["NONE"] * len(gold), sboot, sw))]
    preds = {**{k: v[1] for k, v in detectors.items()}, **{k: oof[k] >= 0.5 for k in ("cal_two", "cal_three", "cal_three_nli")}}
    labels = {**{k: v[0] for k, v in detectors.items()}, **{k: E1_MODELS[k][0] for k in E1_MODELS}}
    for det in ("current", "rel_ctx", "agree", "vote2", "cal_three"):
        for dname, (dlabel, fn) in decision_layers().items():
            pred = [UP[fn(p)] if o else "NONE" for p, o in zip(pairs, preds[det]) if p.status]
            e2e[f"{det}+{dname}"] = four_state(gold, pred, sboot, sw)
            table.append(four_row(f"{labels[det]} → {dlabel}", e2e[f"{det}+{dname}"]))
    res["end_to_end"] = e2e
    lines += [f"End to end, four states, {len(gold)} pairs with a consensus status. The sample holds every system-opposed pair but "
              "only 30 of 97 pairs without any opposition signal, so plain accuracy understates every system and the constant; "
              "the weighted column corrects for that within the labelled pool.", "", md_table(table), ""]

    # oracle decomposition: replace one stage at a time by the reference
    def staged(opp: str, suff: str, comp: str) -> list[str]:
        out = []
        for p in sp:
            ref_opp = p.status != "NONE"
            o = ref_opp if opp == "ref" else p.opposed
            if not o:
                out.append("NONE")
                continue
            v1 = C.v1_status(p.factors)
            insufficient = (p.status == "INSUFFICIENT_EVIDENCE") if (suff == "ref" and ref_opp) else v1 == "insufficient"
            if insufficient:
                out.append("INSUFFICIENT_EVIDENCE")
                continue
            if comp == "ref" and ref_opp and p.status != "INSUFFICIENT_EVIDENCE":
                out.append(p.status)
            elif comp == "cd":
                out.append("CONTEXT_DEPENDENT")
            else:
                out.append("GENUINE" if v1 == "genuine" else "CONTEXT_DEPENDENT")
        return out

    table = [["Opposition", "Sufficiency", "Comparability", "Accuracy", "Macro-F1"]]
    res["oracle"] = {}
    for opp, suff, comp in [("sys", "sys", "sys"), ("ref", "sys", "sys"), ("sys", "ref", "sys"), ("sys", "sys", "ref"), ("sys", "ref", "ref"),
                            ("ref", "ref", "sys"), ("ref", "sys", "ref"), ("ref", "sys", "cd"), ("sys", "sys", "cd")]:
        r = four_state(gold, staged(opp, suff, comp), sboot, sw)
        res["oracle"][f"{opp}/{suff}/{comp}"] = r
        name = {"sys": "system", "ref": "**reference**", "cd": "always context-dep."}
        table.append([name[opp], name[suff], name[comp], pct(r["accuracy"], *r["accuracy_ci"]), pct(r["macro_f1"], *r["macro_f1_ci"])])
    lines += ["Error decomposition: one stage at a time is replaced by the reference answer (where the reference "
              "defines one; the V1 gate supplies the system's sufficiency and comparability answers).", "", md_table(table), ""]

    # relation and status kept apart: what the cached readings look like against the C4 status
    table = [["Reading", *[FS.SHORT[s] for s in FS.STATES], "no consensus"]]
    for label, key in [("stance", lambda p: f"stance: {p.stance}"), ("relation", lambda p: f"relation call: {p.direct['context']}")]:
        for value in sorted({key(p) for p in pairs}):
            c = Counter(p.status for p in pairs if key(p) == value)
            table.append([value, *[str(c.get(s, 0)) for s in FS.STATES], str(c.get(None, 0))])
    res["relation_vs_status"] = table
    lines += ["Relation and conflict status kept apart: each cached reading against the C4 status (raw counts; C4 has "
              "no five-way relation labels, so the relation itself cannot be scored).", "", md_table(table), ""]
    return res, lines, {**sig, **preds, "y": y, "w": w}


# ======================================================================================================= E2
def one_sided(fc) -> bool:
    return fc.status == "unknown" and bool(fc.a) != bool(fc.b)


def stated_both(fc) -> bool:
    return fc.status != "unknown"


FACTOR_VARIANTS: dict[str, tuple[str, Callable[[Pair], set[str]]]] = {
    "llm_material": ("LLM status, LLM materiality (current)", lambda p: {x.factor for x in p.factors if x.status == "different"}),
    "llm_no_materiality": ("LLM status, no materiality flag", lambda p: {x.factor for x in p.factors if x.status == "different" or (x.status == "overlap" and x.note == C.IMMATERIAL_NOTE)}),
    "llm_any": ("LLM status, any stated difference (incl. partial)", lambda p: {x.factor for x in p.factors if x.status in ("different", "overlap")}),
    "deterministic": ("Names and numbers compared in code", lambda p: {x.factor for x in C.with_deterministic_statuses(p.factors) if x.status == "different"}),
    "stated": ("Every factor both papers state", lambda p: {x.factor for x in p.factors if stated_both(x)}),
}


def set_match(sys_sets: list[set[str]], ref_sets: list[set[str]], boot: Boot) -> dict:
    def stats(idx) -> tuple[float, float | None, float | None, float]:
        hit = np.mean([bool(sys_sets[i] & ref_sets[i]) for i in idx])
        inter = sum(len(sys_sets[i] & ref_sets[i]) for i in idx)
        n_sys, n_ref = sum(len(sys_sets[i]) for i in idx), sum(len(ref_sets[i]) for i in idx)
        jac = np.mean([len(sys_sets[i] & ref_sets[i]) / len(sys_sets[i] | ref_sets[i]) if sys_sets[i] | ref_sets[i] else 0.0 for i in idx])
        return float(hit), (inter / n_sys if n_sys else None), (inter / n_ref if n_ref else None), float(jac)

    base = stats(range(len(sys_sets)))
    out = {}
    for k, name in enumerate(("hit", "precision", "recall", "jaccard")):
        out[name] = {"value": base[k], "ci": boot.ci(lambda i, k=k: stats(i)[k])}
    return out


SUFFICIENCY_DEFS: dict[str, tuple[str, Callable[[Pair], float]]] = {
    "E": ("Current count: scope + measurement factors stated by both (paper-level half)", lambda p: p.row.E),
    "both_all": ("All twelve factors stated by both", lambda p: sum(stated_both(x) for x in p.factors)),
    "both_claim": ("Stated by both at claim level only (scope + measurement)", lambda p: sum(stated_both(x) and not x.inherited and C.ROLES[x.factor] in C.SUFFICIENCY_ROLES for x in p.factors)),
    "roles": ("Number of roles with a factor stated by both (0-4)", lambda p: len({C.ROLES[x.factor] for x in p.factors if stated_both(x)})),
    "core": ("Coverage of task + intervention + outcome measure (0-3)", lambda p: sum(stated_both(x) for x in p.factors if x.factor in ("task", "model / method", "evaluation metric"))),
    "one_sided": ("Fewer factors stated by one paper only", lambda p: -sum(one_sided(x) for x in p.factors)),
    "share_both": ("Share of stated factors that both papers state", lambda p: (lambda b, o: b / (b + o) if b + o else 0.0)(sum(stated_both(x) for x in p.factors), sum(one_sided(x) for x in p.factors))),
    "difference": ("A difference was found (evidence of non-comparability is evidence)", lambda p: float(any(x.status == "different" for x in p.factors))),
    "n_same": ("Factors on which the papers agree", lambda p: sum(x.status == "same" for x in p.factors)),
    "values": ("Stated values in total (either paper)", lambda p: sum(bool(x.a) + bool(x.b) for x in p.factors)),
    "claim_len": ("Baseline: length of the shorter claim", lambda p: min(len(p.a.claim_text), len(p.b.claim_text))),
    "sim": ("Baseline: claim similarity", lambda p: p.sim),
}


def mismatch_score(p: Pair, variant: str) -> float:
    """Uniform graded comparability evidence: agreements minus mismatches over the context roles."""
    if variant == "deterministic":
        ev = C.factor_evidence(C.with_deterministic_statuses(p.factors))
    else:
        ev = C.factor_evidence(p.factors, llm_materiality=variant == "llm_material")
    return sum(e.same - e.different for e in ev if e.role in C.SUFFICIENCY_ROLES)


def run_e2(pairs: list[Pair], boot: Boot, e1: dict[str, np.ndarray], n_perm: int, seed: int) -> tuple[dict, list[str]]:
    rng = np.random.default_rng(seed)
    res: dict = {}
    lines = ["## E2. Factor layer and evidence sufficiency", ""]
    topics = np.array([p.topic for p in pairs])

    # ---- (a) does the table name the factors the annotators name?
    cd = [p for p in pairs if p.status == "CONTEXT_DEPENDENT"]
    cboot = Boot([p.cluster for p in cd], len(boot.samples), seed + 1)
    role_of = lambda s: {C.ROLES[x] for x in s}  # noqa: E731
    lines += [f"### (a) Explanatory match on the {len(cd)} CONTEXT_DEPENDENT pairs", "",
              "System set: the factors a representation marks as differing. Reference set: factors named by at least one "
              "annotator (and, in the last column, by at least two). Hit = the two sets share a factor.", ""]
    res["match"] = {}
    for level, conv in (("factor", lambda s: set(s)), ("role", role_of)):
        ref1 = [conv(set(p.ref["factor_votes"])) for p in cd]
        ref2 = [conv(set(p.ref["factors"])) for p in cd]
        freq = Counter(x for s in ref1 for x in s)
        top = freq.most_common(1)[0][0]
        table = [[f"Representation ({level} level)", "Mean set size", "Hit", "Precision", "Recall", "Jaccard", "Shuffle p (hit)", "Hit, ≥2 annotators"]]
        variants = {**{k: (lab, [conv(fn(p)) for p in cd]) for k, (lab, fn) in FACTOR_VARIANTS.items()},
                    "constant": (f"Baseline: always \"{top}\"", [{top} for _ in cd])}
        for name, (label, sets) in variants.items():
            m = set_match(sets, ref1, cboot)
            m2 = set_match(sets, ref2, cboot)
            perm_p = None
            if name != "constant":
                ct = np.array([p.topic for p in cd])
                obs = m["hit"]["value"]
                ge = 0
                for _ in range(n_perm):
                    order = np.arange(len(cd))
                    for t in np.unique(ct):
                        idx = np.flatnonzero(ct == t)
                        order[idx] = rng.permutation(idx)
                    ge += np.mean([bool(sets[order[i]] & ref1[i]) for i in range(len(cd))]) >= obs - 1e-12
                perm_p = (ge + 1) / (n_perm + 1)
            res["match"][f"{level}:{name}"] = {**m, "mean_size": float(np.mean([len(s) for s in sets])), "shuffle_p": perm_p, "hit_two_annotators": m2["hit"]}
            table.append([label, f(float(np.mean([len(s) for s in sets])), 1), cell(m["hit"]), cell(m["precision"]), cell(m["recall"]),
                          f(m["jaccard"]["value"]), f(perm_p, 3), cell(m2["hit"])])
        lines += [md_table(table), ""]
    # agreement among the annotators themselves, factor vs role level
    def ann_jaccard(conv) -> float:
        vals = []
        for p in cd:
            sets = [conv(set(p.ref["labels"][a]["factors"])) for a in ANNOTATORS if p.ref["labels"][a]["status"] == "CONTEXT_DEPENDENT"]
            vals += [len(x & y) / len(x | y) for i, x in enumerate(sets) for y in sets[i + 1:] if x | y]
        return float(np.mean(vals)) if vals else float("nan")

    res["annotator_jaccard"] = {"factor": ann_jaccard(lambda s: set(s)), "role": ann_jaccard(role_of)}
    counts = Counter(x for p in cd for x in p.ref["factor_votes"])
    roles = Counter(r for p in cd for r in role_of(set(p.ref["factor_votes"])))
    res["annotator_factor_counts"] = dict(counts)
    res["annotator_role_counts"] = dict(roles)
    lines += [f"Annotators against each other on the same pairs (both called it CONTEXT_DEPENDENT): mean pairwise Jaccard "
              f"{res['annotator_jaccard']['factor']:.2f} at factor level, {res['annotator_jaccard']['role']:.2f} at role level. "
              f"Pairs in which a role is named by at least one annotator: " + ", ".join(f"{k} {v}" for k, v in roles.most_common())
              + ". Factors: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()) + ".", ""]

    # ---- (b) does the table add signal beyond the semantic layer?
    y = e1["y"]
    sem = np.c_[tuple(np.array(v, float) for v in (np.array([p.stance == "oppose" for p in pairs]), e1["rel_ctx"], e1["rel_claims"]))]
    fac = np.array([[sum(e.different for e in ev if e.role in C.SUFFICIENCY_ROLES), sum(e.same for e in ev if e.role in C.SUFFICIENCY_ROLES),
                     float(any(e.different > 0 for e in ev if e.role == "intervention")), sum(e.known for e in ev)]
                    for ev in (C.factor_evidence(p.factors) for p in pairs)])
    p_sem, p_both, p_fac = loto(sem, y, topics), loto(np.c_[sem, fac], y, topics), loto(fac, y, topics)
    d_auc = (S.auc(y.astype(float), p_both) or 0) - (S.auc(y.astype(float), p_sem) or 0)
    lo, hi = boot.ci(lambda i: (S.auc(y[i].astype(float), p_both[i]) or 0) - (S.auc(y[i].astype(float), p_sem[i]) or 0))
    res["beyond_semantic"] = {"auc_semantic": S.auc(y.astype(float), p_sem), "auc_semantic_plus_factors": S.auc(y.astype(float), p_both),
                              "auc_factors_only": S.auc(y.astype(float), p_fac), "delta_auc": d_auc, "delta_ci": [lo, hi],
                              "log_loss": {"semantic": S.log_loss(y.astype(float), p_sem), "plus_factors": S.log_loss(y.astype(float), p_both), "factors_only": S.log_loss(y.astype(float), p_fac)}}
    lines += ["### (b) Signal beyond the semantic layer", "",
              "Opposition (all pairs), leave-one-topic-out. Factor features: context mismatches, context agreements, an "
              "intervention mismatch, number of factors stated by both.", "",
              md_table([["Score", "AUC", "Log-loss"],
                        ["Semantic readings only (stance + two relation calls)", auc_cell(y, p_sem, boot), f(S.log_loss(y.astype(float), p_sem), 3)],
                        ["Semantic + factor features", auc_cell(y, p_both, boot), f(S.log_loss(y.astype(float), p_both), 3)],
                        ["Factor features only", auc_cell(y, p_fac, boot), f(S.log_loss(y.astype(float), p_fac), 3)]]),
              "", f"Adding the factor features changes the AUC by {d_auc:+.3f} [{lo:+.3f}, {hi:+.3f}].", ""]

    # GENUINE vs CONTEXT_DEPENDENT, by materiality variant (underpowered; descriptive)
    gc = [p for p in pairs if p.status in ("GENUINE", "CONTEXT_DEPENDENT")]
    gboot = Boot([p.cluster for p in gc], len(boot.samples), seed + 2)
    yg = np.array([p.status == "GENUINE" for p in gc])
    soft = [p for p in pairs if p.ref["stance"] == "oppose"]
    sboot2 = Boot([p.cluster for p in soft], len(boot.samples), seed + 3)
    ys = np.array([any(p.ref["labels"][a]["status"] == "GENUINE" for a in ANNOTATORS) for p in soft])
    table = [["Uniform score (agreements − mismatches over context factors)", f"AUC, consensus GENUINE ({int(yg.sum())} of {len(gc)})",
              f"AUC, any annotator GENUINE ({int(ys.sum())} of {len(soft)} opposed pairs)"]]
    res["materiality"] = {}
    for variant, label in (("llm_material", "LLM materiality flag used (current)"), ("none", "No materiality flag"), ("deterministic", "Names and numbers compared in code")):
        sg = np.array([mismatch_score(p, variant) for p in gc])
        ss = np.array([mismatch_score(p, variant) for p in soft])
        res["materiality"][variant] = {"consensus": auc_ci(yg, sg, gboot), "any_annotator": auc_ci(ys, ss, sboot2)}
        table.append([label, auc_cell(yg, sg, gboot), auc_cell(ys, ss, sboot2)])
    for role in C.ROLE_ORDER:
        sg = np.array([-sum(e.different for e in C.factor_evidence(p.factors) if e.role == role) for p in gc])
        ss = np.array([-sum(e.different for e in C.factor_evidence(p.factors) if e.role == role) for p in soft])
        table.append([f"Mismatches in role \"{role}\" only", auc_cell(yg, sg, gboot), auc_cell(ys, ss, sboot2)])
    lines += ["GENUINE against CONTEXT_DEPENDENT. With this few genuine pairs no row can be told from another; the table "
              "is descriptive.", "", md_table(table), ""]

    # ---- (c) evidence sufficiency
    opp = [p for p in pairs if p.status and p.status != "NONE"]
    oboot = Boot([p.cluster for p in opp], len(boot.samples), seed + 4)
    yi = np.array([p.status == "INSUFFICIENT_EVIDENCE" for p in opp])
    ro = [p for p in pairs if p.ref["stance"] == "oppose"]
    roboot = Boot([p.cluster for p in ro], len(boot.samples), seed + 5)
    y_any = np.array([any(p.ref["labels"][a]["sufficiency"] == "insufficient" for a in ANNOTATORS) for p in ro])
    y_dim = np.array([p.ref["sufficiency"] == "insufficient" for p in pairs])
    table = [["Definition (higher = more evidence)", f"AUC, status INSUFFICIENT ({int(yi.sum())} of {len(opp)} opposed pairs)",
              f"AUC, any annotator \"insufficient\" ({int(y_any.sum())} of {len(ro)})", f"AUC, dimension B, all pairs ({int(y_dim.sum())} of {len(pairs)})"]]
    res["sufficiency"] = {}
    for name, (label, fn) in SUFFICIENCY_DEFS.items():
        cells = []
        res["sufficiency"][name] = {}
        for tag, ps, yy, bb in (("status", opp, yi, oboot), ("any", ro, y_any, roboot), ("dimension_b", pairs, y_dim, boot)):
            s = -np.array([float(fn(p)) for p in ps])  # low evidence should rank as insufficient
            res["sufficiency"][name][tag] = auc_ci(yy, s, bb)
            cells.append(auc_cell(yy, s, bb))
        table.append([label, *cells])
    lines += ["### (c) Evidence sufficiency", "",
              "Each definition is fixed in advance and nothing is fitted. An AUC above 0.5 means that pairs with less of "
              "that evidence are the ones annotators call insufficient.", "", md_table(table), ""]
    kappa = {a: Counter(p.ref["labels"][a]["status"] for p in ro)["INSUFFICIENT_EVIDENCE"] for a in ANNOTATORS}
    lines += [f"The target itself is unstable: on the {len(ro)} pairs opposed by majority, the three annotators call "
              + ", ".join(f"{v} ({k})" for k, v in kappa.items()) + " pairs INSUFFICIENT.", ""]
    res["insufficient_by_annotator"] = kappa

    # ---- (d) ablations of the two count-based definitions
    def count(p: Pair, keep: Callable[[object], bool], weighted: bool) -> float:
        return sum((0.5 if (weighted and x.inherited) else 1.0) for x in p.factors if stated_both(x) and keep(x))

    table = [["Count of factors stated by both, restricted to", "AUC, status INSUFFICIENT", "AUC, any annotator \"insufficient\""]]
    res["ablation"] = {}
    specs: list[tuple[str, Callable[[object], bool]]] = [("all twelve factors", lambda x: True)]
    specs += [(f"role {r} only", lambda x, r=r: C.ROLES[x.factor] == r) for r in C.ROLE_ORDER]
    specs += [(f"without role {r}", lambda x, r=r: C.ROLES[x.factor] != r) for r in C.ROLE_ORDER]
    specs += [(f"without \"{k}\"", lambda x, k=k: x.factor != k) for k in C.FACTOR_ORDER]
    for label, keep in specs:
        s1 = -np.array([count(p, keep, False) for p in opp])
        s2 = -np.array([count(p, keep, False) for p in ro])
        res["ablation"][label] = {"status": auc_ci(yi, s1, oboot), "any": auc_ci(y_any, s2, roboot)}
        table.append([label, auc_cell(yi, s1, oboot), auc_cell(y_any, s2, roboot)])
    lines += ["### (d) Factor and role ablations", "", md_table(table), ""]
    never = [k for k in C.FACTOR_ORDER if not any(stated_both(x) for p in pairs for x in p.factors if x.factor == k)]
    usage = Counter(x.factor for p in pairs for x in p.factors if stated_both(x))
    res["stated_by_both"] = dict(usage)
    lines += ["Pairs in which both papers state a factor: " + ", ".join(f"{k} {usage.get(k, 0)}" for k in C.FACTOR_ORDER)
              + (f". Never stated by both: {', '.join(never)}." if never else "."), ""]
    return res, lines


# ======================================================================================================= E3
EFFECT_VERB = {"positive": "improves", "negative": "worsens", "none": "does not change", "mixed": "has a mixed effect on"}


def proposition(fr: FR.ClaimFrame) -> str:
    """A frame rendered as one short sentence, for the entailment model."""
    tail = f" compared with {fr.comparator}" if fr.comparator else ""
    return f"{fr.intervention} {EFFECT_VERB.get(fr.effect, 'is related to')} {fr.outcome}{tail}."


class PhraseSim:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache: dict[str, np.ndarray] = {}

    def prepare(self, phrases: list[str]) -> None:
        new = sorted({p for p in phrases if p and p not in self.cache})
        if new:
            vecs = np.asarray(self.embedder.embed(new), float)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            self.cache.update(zip(new, vecs))

    def __call__(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        self.prepare([a, b])
        return float(np.clip(self.cache[a] @ self.cache[b], 0.0, 1.0))


THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


def run_e3(pairs: list[Pair], boot: Boot, sboot: Boot, e1: dict[str, np.ndarray], batch: str, seed: int) -> tuple[dict, list[str]]:
    frames = FX.load_frames(batch)
    raw = FX.load_raw(batch)
    models = get_models(Settings())
    sim = PhraseSim(models.embedder)
    key = lambda p, cid: (p.ref["run"], p.topic, cid)  # noqa: E731
    have = np.array([key(p, p.rec["a"]) in frames and key(p, p.rec["b"]) in frames for p in pairs])
    fa = [frames.get(key(p, p.rec["a"])) for p in pairs]
    fb = [frames.get(key(p, p.rec["b"])) for p in pairs]
    sim.prepare([x for fr in frames.values() for x in (fr.intervention, fr.comparator, fr.outcome)])
    cmp = [FR.compare_frames(a, b, sim) if h else None for a, b, h in zip(fa, fb, have)]
    cmp_noswap = [FR.compare_frames(a, b, sim, allow_swap=False) if h else None for a, b, h in zip(fa, fb, have)]
    y, w = e1["y"], e1["w"]
    topics = np.array([p.topic for p in pairs])
    res: dict = {"frames": {}}
    lines = ["## E3. Conditional-prediction frames", ""]

    # ---- extraction
    fl = list(frames.values())
    n_cond_raw = sum(len(o.get("conditions") or []) for o in raw.values())
    n_cond = sum(len(fr.conditions) for fr in fl)
    levels = Counter(c.level for fr in fl for c in fr.conditions.values())
    stated = [fr for fr in fl if fr.effect != "not_stated"]
    res["frames"] = {"claims": len(raw), "valid": len(fl), "effects": dict(Counter(fr.effect for fr in fl)),
                     "effect_quote_in_claim": sum(fr.effect_grounded for fr in stated), "with_effect": len(stated),
                     "new_method": sum(fr.is_new for fr in fl), "conditional": sum(fr.conditional for fr in fl),
                     "conditions_raw": n_cond_raw, "conditions_grounded": n_cond, "condition_levels": dict(levels),
                     "pairs_with_both_frames": int(have.sum())}
    lines += [f"Extraction: {len(raw)} claims, one call each; {len(fl)} valid frames. Effects: "
              + ", ".join(f"{k} {v}" for k, v in Counter(fr.effect for fr in fl).most_common())
              + f". The effect quote is found in the claim sentence for {sum(fr.effect_grounded for fr in stated)} of {len(stated)} frames with an effect. "
              f"{sum(fr.is_new for fr in fl)} frames are about a method the paper itself proposes; {sum(fr.conditional for fr in fl)} claims restrict themselves to conditions. "
              f"Conditions: {n_cond} of {n_cond_raw} values survive the quote check ("
              + ", ".join(f"{k} {v}" for k, v in levels.most_common()) + f"). Both frames exist for {int(have.sum())} of {len(pairs)} pairs.", ""]

    # ---- component signals against the reference opposition
    feat = {
        "both_effect": np.array([bool(c and c.both_effect) for c in cmp]),
        "incompatible": np.array([bool(c and c.incompatible) for c in cmp]),
        "new_mismatch": np.array([bool(c and c.new_mismatch) for c in cmp]),
        "s_intervention": np.array([c.s_intervention if c else 0.0 for c in cmp]),
        "s_outcome": np.array([c.s_outcome if c else 0.0 for c in cmp]),
    }
    so = e1["current"]
    sub = Boot([p.cluster for p in pairs if p.opposed], len(boot.samples), seed + 6)
    table = [["Frame component", f"AUC, all {len(pairs)} pairs", f"AUC, {int(so.sum())} system-opposed pairs"]]
    for name, sign in (("both_effect", 1), ("incompatible", 1), ("new_mismatch", -1), ("s_intervention", 1), ("s_outcome", 1)):
        s = sign * feat[name].astype(float)
        table.append([name + (" (reversed)" if sign < 0 else ""), auc_cell(y, s, boot), auc_cell(y[so], s[so], sub)])
    eff = Counter(((c.effects or "no effect stated") if c else "no frame", bool(yy)) for c, yy in zip(cmp, y))
    lines += [md_table(table), "", "Effect relation of the two frames against the reference (opposed / not opposed): "
              + "; ".join(f"{k}: {eff.get((k, True), 0)} / {eff.get((k, False), 0)}" for k in sorted({k for k, _ in eff})) + ".", ""]
    res["components"] = {name: auc_ci(y, feat[name].astype(float), boot) for name in feat}
    res["effect_relation"] = {f"{k}|{'opposed' if v else 'not'}": n for (k, v), n in eff.items()}

    # ---- detectors
    def rule(t: float, idx=None) -> np.ndarray:
        return np.array([bool(c) and FR.frame_relation(c, t) == "CONTRADICTS" for c in cmp])

    rule_oof = np.zeros(len(pairs), dtype=bool)
    chosen = {}
    for t in np.unique(topics):
        test = topics == t
        best = max(THRESHOLDS, key=lambda th: (prf(y[~test], rule(th)[~test], w[~test])["f1"] or 0, -th))
        chosen[str(t)] = best
        rule_oof[test] = rule(best)[test]
    veto = np.array([bool(c) and (not c.both_effect or c.new_mismatch or c.effects == "agree") for c in cmp])
    Xf = np.c_[tuple(feat[k].astype(float) for k in ("both_effect", "incompatible", "new_mismatch", "s_intervention", "s_outcome"))]
    sem = np.c_[tuple(np.array(v, float) for v in (np.array([p.stance == "oppose" for p in pairs]), e1["rel_ctx"], e1["rel_claims"]))]
    p_frames, p_sem, p_comb = loto(Xf, y, topics), loto(sem, y, topics), loto(np.c_[sem, Xf], y, topics)
    detectors = {
        "current": ("Current trigger", so),
        "agree": ("E1: both readings agree", e1["agree"]),
        "frame_rule0": ("Frames, no similarity threshold: both state an effect, effects incompatible, no unnamed new method", rule(0.0)),
        "frame_rule": ("Frames, same rule with a similarity threshold chosen leave-one-topic-out", rule_oof),
        "frame_cal": ("Frames, calibrated score (leave-one-topic-out), p ≥ 0.5", p_frames >= 0.5),
        "current_veto": ("Current trigger unless the frames veto (no effect stated, unnamed new method, or same effect)", so & ~veto),
        "agree_veto": ("E1 both-agree unless the frames veto", e1["agree"] & ~veto),
        "comb_cal": ("Calibrated: stance + relation calls + frame features, p ≥ 0.5", p_comb >= 0.5),
        "frame_rule0_noswap": ("POST HOC — frames, no threshold, interventions always aligned directly (no role swap)",
                               np.array([bool(c) and FR.frame_relation(c, 0.0) == "CONTRADICTS" for c in cmp_noswap])),
        "frame_rule0_noswap_mixed": ("POST HOC — as above, and \"it depends\" against a definite effect also counts as opposed",
                                     np.array([bool(c) and FR.frame_relation(c, 0.0) in ("CONTRADICTS", "QUALIFIES") for c in cmp_noswap])),
    }
    table = [["Detector", "Flagged / right", "Precision", "Recall", "F1"]]
    res["detectors"] = {}
    for name, (label, pred) in detectors.items():
        m = binary(y, pred, w, boot)
        res["detectors"][name] = m
        table.append([label, f"{m['flagged']} / {m['correct']}", cell(m["precision"]), cell(m["recall"]), cell(m["f1"])])
    d_auc = (S.auc(y.astype(float), p_comb) or 0) - (S.auc(y.astype(float), p_sem) or 0)
    lo, hi = boot.ci(lambda i: (S.auc(y[i].astype(float), p_comb[i]) or 0) - (S.auc(y[i].astype(float), p_sem[i]) or 0))
    res["calibrated"] = {"auc_frames": auc_ci(y, p_frames, boot), "auc_semantic": auc_ci(y, p_sem, boot), "auc_combined": auc_ci(y, p_comb, boot),
                         "delta_combined_minus_semantic": [d_auc, lo, hi], "thresholds_by_heldout_topic": chosen,
                         "coefficients_frames": full_fit(Xf, y)}
    lines += ["Opposition detection from frames (target and weighting as in E1):", "", md_table(table), "",
              f"Calibrated scores, out of fold: frames only AUC {auc_cell(y, p_frames, boot)}; semantic readings only {auc_cell(y, p_sem, boot)}; "
              f"both {auc_cell(y, p_comb, boot)}; adding the frame features changes the AUC by {d_auc:+.3f} [{lo:+.3f}, {hi:+.3f}]. "
              f"Similarity thresholds chosen per held-out topic: {sorted({float(v) for v in chosen.values()})}. "
              f"Alignment used: {dict(Counter(c.alignment for c in cmp if c))}. The two rows marked POST HOC were added after the "
              "pre-specified rows had been read, to see whether the role-swap rule or the \"mixed\" category explains the failure; "
              "they are not evidence for a design.", ""]

    # ---- which frame test removes what
    def reason(c) -> str:
        if c is None:
            return "no frame"
        if not c.both_effect:
            return "a claim states no effect"
        if c.new_mismatch:
            return "new method not named by the other claim"
        return {"agree": "same effect", "mixed": "a claim says \"it depends\"", "opposite": "kept: opposite effects", "effect_vs_none": "kept: effect vs none"}[c.effects]

    table = [["Frame verdict", "False opposed (system opposed, reference not)", "True opposed (both)", "Missed by the system (reference opposed)"]]
    groups = {"false": [c for c, p, yy in zip(cmp, pairs, y) if p.opposed and not yy], "true": [c for c, p, yy in zip(cmp, pairs, y) if p.opposed and yy],
              "missed": [c for c, p, yy in zip(cmp, pairs, y) if not p.opposed and yy]}
    counts = {g: Counter(reason(c) for c in cs) for g, cs in groups.items()}
    for r in sorted({r for c in counts.values() for r in c}, key=lambda r: (r.startswith("kept"), r)):
        table.append([r, *[str(counts[g].get(r, 0)) for g in ("false", "true", "missed")]])
    table.append(["total", *[str(len(groups[g])) for g in ("false", "true", "missed")]])
    res["verdicts"] = {g: dict(c) for g, c in counts.items()}
    lines += ["What the frames say about the system's errors (no similarity threshold):", "", md_table(table), ""]

    # ---- entailment model on normalised propositions (free arm)
    def nli_con(texts: list[tuple[str, str]]) -> np.ndarray:
        out = models.nli.predict([t for a, b in texts for t in ((a, b), (b, a))])
        return np.array([(out[2 * i]["contradiction"] + out[2 * i + 1]["contradiction"]) / 2 for i in range(len(texts))])

    concl = nli_con([(str(p.rec["llm_pair"].get("conclusion_a") or p.a.claim_text), str(p.rec["llm_pair"].get("conclusion_b") or p.b.claim_text)) for p in pairs])
    props = nli_con([(proposition(a), proposition(b)) if h else (p.a.claim_text, p.b.claim_text) for a, b, h, p in zip(fa, fb, have, pairs)])
    table = [["Entailment model input (mean contradiction score)", f"AUC, all {len(pairs)} pairs", f"AUC, {int(so.sum())} system-opposed pairs"],
             ["Raw claim sentences (current)", auc_cell(y, e1["nli_con"], boot), auc_cell(y[so], e1["nli_con"][so], sub)],
             ["Cached one-clause conclusions of the pair analysis", auc_cell(y, concl, boot), auc_cell(y[so], concl[so], sub)],
             ["Frames rendered as one sentence each", auc_cell(y, props, boot), auc_cell(y[so], props[so], sub)]]
    res["nli_inputs"] = {"raw": auc_ci(y, e1["nli_con"], boot), "conclusions": auc_ci(y, concl, boot), "frame_propositions": auc_ci(y, props, boot)}
    yf = y.astype(float)
    d_c = (S.auc(yf, concl) or 0) - (S.auc(yf, e1["nli_con"]) or 0)
    dlo, dhi = boot.ci(lambda i: (S.auc(yf[i], concl[i]) or 0) - (S.auc(yf[i], e1["nli_con"][i]) or 0))
    p_sem_c = loto(np.c_[sem, concl], y, topics)
    a_c = (S.auc(yf, p_sem_c) or 0) - (S.auc(yf, p_sem) or 0)
    alo, ahi = boot.ci(lambda i: (S.auc(yf[i], p_sem_c[i]) or 0) - (S.auc(yf[i], p_sem[i]) or 0))
    res["nli_inputs"]["conclusions_minus_raw"] = [d_c, dlo, dhi]
    res["nli_inputs"]["added_to_semantic"] = [a_c, alo, ahi]
    lines += [f"The same local entailment model ({type(models.nli).__name__}) on three representations of the claims:", "", md_table(table), "",
              f"Conclusions minus raw sentences: {d_c:+.2f} [{dlo:+.2f}, {dhi:+.2f}]. The conclusions were written by the pair analysis, which had "
              f"read both claims, so this is not an independent reading: added to the calibrated semantic score (stance + two relation calls) it "
              f"changes the out-of-fold AUC by {a_c:+.3f} [{alo:+.3f}, {ahi:+.3f}].", ""]

    # ---- conflict status from frames
    gold = [p.status for p in pairs if p.status]
    thr = np.array([chosen[str(p.topic)] for p in pairs])
    status_rule = [UP[FR.frame_status(c, t)] if c else "NONE" for c, t in zip(cmp, thr)]
    status0 = [UP[FR.frame_status(c, 0.0)] if c else "NONE" for c in cmp]

    def frame_decision(c) -> str:
        if c is None or not c.shared and not c.conditional:
            return "insufficient"
        return "context_dependent" if (any(s == "different" for s in c.shared.values()) or c.conditional) else "genuine"

    sw = np.array([STRATUM_WEIGHT.get(p.stratum, 1.0) for p in pairs if p.status])
    table = [FOUR_HEAD, four_row("Constant NONE", four_state(gold, ["NONE"] * len(gold), sboot, sw)),
             four_row("Current ClaimGraph (trigger → V1 gate)", four_state(gold, [UP[C.v1_status(p.factors)] if p.opposed else "NONE" for p in pairs if p.status], sboot, sw))]
    res["end_to_end"] = {}
    for name, label, pred in [
        ("frames", "Frames end to end (threshold leave-one-topic-out)", status_rule),
        ("frames0", "Frames end to end (no similarity threshold)", status0),
        ("agree_frames", "E1 both-agree → frame conditions", [UP[frame_decision(c)] if o else "NONE" for c, o in zip(cmp, e1["agree"])]),
        ("agree_veto_frames", "E1 both-agree unless frames veto → frame conditions", [UP[frame_decision(c)] if o else "NONE" for c, o in zip(cmp, detectors["agree_veto"][1])]),
        ("agree_veto_cd", "E1 both-agree unless frames veto → always CONTEXT_DEPENDENT", ["CONTEXT_DEPENDENT" if o else "NONE" for o in detectors["agree_veto"][1]]),
        ("agree_veto_v1", "E1 both-agree unless frames veto → V1 gate", [UP[C.v1_status(p.factors)] if o else "NONE" for p, o in zip(pairs, detectors["agree_veto"][1])]),
    ]:
        r = four_state(gold, [x for x, p in zip(pred, pairs) if p.status], sboot, sw)
        res["end_to_end"][name] = r
        table.append(four_row(label, r))
    lines += [f"End to end, four states, {len(gold)} pairs with a consensus status:", "", md_table(table), ""]

    # the decision layer alone, on pairs the reference calls opposed
    opp = [(p, c) for p, c in zip(pairs, cmp) if p.status and p.status != "NONE"]
    oboot = Boot([p.cluster for p, _ in opp], len(boot.samples), seed + 7)
    og = [p.status for p, _ in opp]
    table = [["Decision layer on the reference-opposed pairs", "Accuracy", "GENUINE said / right", "INSUFFICIENT said / right"]]
    res["decision_layer"] = {}
    for name, label, pred in [("cd", "Always CONTEXT_DEPENDENT", ["CONTEXT_DEPENDENT"] * len(opp)),
                              ("v1", "V1 gate on the pair-analysis table", [UP[C.v1_status(p.factors)] for p, _ in opp]),
                              ("frames", "Frame conditions", [UP[frame_decision(c)] for _, c in opp])]:
        r = four_state(og, pred, oboot)
        res["decision_layer"][name] = r
        table.append([label, pct(r["accuracy"], *r["accuracy_ci"]), f"{r['GENUINE']['pred']} / {r['GENUINE']['tp']}", f"{r['INSUFFICIENT_EVIDENCE']['pred']} / {r['INSUFFICIENT_EVIDENCE']['tp']}"])
    yi = np.array([p.status == "INSUFFICIENT_EVIDENCE" for p, _ in opp])
    n_shared = np.array([len(c.shared) if c else 0 for _, c in opp], float)
    n_one = np.array([len(c.one_sided) if c else 0 for _, c in opp], float)
    cond = np.array([bool(c and c.conditional) for _, c in opp], float)
    res["sufficiency"] = {"shared_conditions": auc_ci(yi, -n_shared, oboot), "one_sided_conditions": auc_ci(yi, n_one, oboot), "conditional_claim": auc_ci(yi, -cond, oboot)}
    lines += [md_table(table), "",
              f"INSUFFICIENT against the frame conditions on the same {len(opp)} pairs (AUC): fewer shared conditions {auc_cell(yi, -n_shared, oboot)}; "
              f"more one-sided conditions {auc_cell(yi, n_one, oboot)}; neither claim conditional {auc_cell(yi, -cond, oboot)}.", ""]

    # do the frame conditions name the annotators' factors?
    cd = [(p, c) for p, c in zip(pairs, cmp) if p.status == "CONTEXT_DEPENDENT" and c]
    cboot = Boot([p.cluster for p, _ in cd], len(boot.samples), seed + 8)
    ref1 = [set(p.ref["factor_votes"]) for p, _ in cd]
    table = [["Frame-based factor set on CONTEXT_DEPENDENT pairs", "Mean size", "Hit", "Precision", "Recall"]]
    res["match"] = {}
    for name, label, sets in [
        ("different", "Conditions both claims state and that differ", [{k for k, s in c.shared.items() if s == "different"} for _, c in cd]),
        ("claim_level", "Conditions quoted from a claim sentence itself", [{k for fr in (frames[key(p, p.rec["a"])], frames[key(p, p.rec["b"])]) for k, v in fr.conditions.items() if v.level == "claim"} for p, _ in cd]),
        ("pair_table", "Pair-analysis table, current (for comparison)", [{x.factor for x in p.factors if x.status == "different"} for p, _ in cd]),
    ]:
        m = set_match(sets, ref1, cboot)
        res["match"][name] = m
        table.append([label, f(float(np.mean([len(s) for s in sets])), 1), cell(m["hit"]), cell(m["precision"]), cell(m["recall"])])
        res["match"][name]["sets"] = [sorted(s) for s in sets]
    top = Counter(x for s in ref1 for x in s).most_common(1)[0][0]
    m = set_match([{top} for _ in cd], ref1, cboot)
    table.append([f"Baseline: always \"{top}\"", "1.0", cell(m["hit"]), cell(m["precision"]), cell(m["recall"])])
    cl, pt = [set(x) for x in res["match"]["claim_level"]["sets"]], [set(x) for x in res["match"]["pair_table"]["sets"]]

    def pr(sets, idx) -> float:
        n = sum(len(sets[i]) for i in idx)
        return sum(len(sets[i] & ref1[i]) for i in idx) / n if n else 0.0

    def rc(sets, idx) -> float:
        n = sum(len(ref1[i]) for i in idx)
        return sum(len(sets[i] & ref1[i]) for i in idx) / n if n else 0.0

    full = range(len(cd))
    dp, dr = pr(cl, full) - pr(pt, full), rc(cl, full) - rc(pt, full)
    plo, phi = cboot.ci(lambda i: pr(cl, i) - pr(pt, i))
    rlo, rhi = cboot.ci(lambda i: rc(cl, i) - rc(pt, i))
    res["match"]["claim_level_minus_pair_table"] = {"precision": [dp, plo, phi], "recall": [dr, rlo, rhi]}
    lines += [md_table(table), "", f"Claim-sentence conditions minus the pair table: precision {100 * dp:+.1f} [{100 * plo:+.0f}, {100 * phi:+.0f}], "
              f"recall {100 * dr:+.1f} [{100 * rlo:+.0f}, {100 * rhi:+.0f}] percentage points.", ""]
    return res, lines


# ====================================================================================================== driver
def run_study(batch: str = "c4", out_dir: Path | None = None, *, n_boot: int = 2000, n_perm: int = 1000, seed: int = 23,
              parts: tuple[str, ...] = ("e1", "e2", "e3")) -> dict:
    pairs = load_pairs(batch)
    boot = Boot([p.cluster for p in pairs], n_boot, seed)
    scored = np.array([p.status is not None for p in pairs])
    sboot = Boot([p.cluster for p in pairs if p.status], n_boot, seed + 100)
    head = ["# Layer study on the C4 four-state labels (E1-E3)", "",
            f"{len(pairs)} pairs from {len({p.cluster for p in pairs})} paper pairs and {len({p.topic for p in pairs})} topics; "
            f"{int(scored.sum())} have a consensus status "
            + "(" + ", ".join(f"{FS.SHORT[s]} {sum(p.status == s for p in pairs)}" for s in FS.STATES) + "). "
            "Reference: blinded model-assisted consensus labels, unadjudicated. Development data: every number below "
            "was computed on topics that have been looked at, and nothing here is a held-out result. Intervals are 95% "
            "bootstrap intervals over paper pairs. With 4 GENUINE pairs no system can be validated on that class.", ""]
    res: dict = {"batch": batch, "n_pairs": len(pairs), "n_scored": int(scored.sum()), "n_boot": n_boot, "seed": seed}
    r1, l1, e1 = run_e1(pairs, boot, sboot, scored)
    res["e1"] = r1
    body = l1
    if "e2" in parts:
        res["e2"], l2 = run_e2(pairs, boot, e1, n_perm, seed)
        body += l2
    if "e3" in parts:
        res["e3"], l3 = run_e3(pairs, boot, sboot, e1, batch, seed)
        body += l3
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "layer_study.md").write_text("\n".join(head + body) + "\n")
        (out_dir / "layer_study.json").write_text(json.dumps(res, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    return res
