"""Four-state development study: opposition, evidence sufficiency and comparability, tested separately.

Reference: unadjudicated model consensus under the four-state protocol (``conflict_annotation``); all of it is
development data.

Table A  evidence sufficiency   can the system's shared-context count tell INSUFFICIENT from sufficient?
Table B  comparability          among opposed pairs with sufficient evidence, can the factor evidence tell
                                GENUINE from CONTEXT_DEPENDENT?

Then the end-to-end four-state decision, the "any shared context" retest, and whether the factor table names the
factors the annotators hold responsible.
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
from app.evaluation import comparability_study as S
from app.evaluation import conflict_annotation as CA
from app.evaluation.experiment import V2_PARAMS, Systems, direct_label, llm_view, load_topic, predict_full
from app.evaluation.scoring import fisher_exact_two_sided
from app.reasoning import comparability as C
from app.schemas.domain import FactorComparison

STATES = ["NONE", "GENUINE", "CONTEXT_DEPENDENT", "INSUFFICIENT_EVIDENCE"]
SHORT = {"NONE": "none", "GENUINE": "genuine", "CONTEXT_DEPENDENT": "context-dep.", "INSUFFICIENT_EVIDENCE": "insufficient"}
ANNOTATOR_IDS = ["gemini", "qwen", "haiku"]
f, md_table = S.f, S.md_table


@dataclass
class Row:
    item_id: str
    pair_id: str
    topic: str
    stratum: str
    ref: dict  # consensus labels and the three annotators' labels
    opposed: bool  # system: face-value conflict (V1 trigger)
    stance: str | None
    con: float
    ent: float
    sim: float
    nli_label: str
    direct: dict[str, str | None]
    old_label: str | None  # forced-choice relation of the first protocol, if the pair was in that batch
    factors: list[FactorComparison] = field(default_factory=list)

    @property
    def status(self) -> str | None:
        return self.ref["status"]

    @property
    def E(self) -> float:
        return C.sufficiency(S._ev(self))  # type: ignore[arg-type]


def load_rows(batch: str = "c4") -> list[Row]:
    d = EVAL_DIR / "annotation" / batch
    ref = json.loads((d / "reference.json").read_text())["items"]
    old_dir = EVAL_DIR / "annotation" / "eval2"
    old_gold = json.loads((old_dir / "gold.json").read_text())["items"]
    old = {k["pair_id"]: old_gold[k["item_id"]]["final"] for k in json.loads((old_dir / "eval2_key.json").read_text())["items"]}
    datas, systems, records = {}, {}, {}
    rows = []
    for iid, r in ref.items():
        key = (r["run"], r["topic"])
        if key not in datas:
            datas[key] = load_topic(*key)
            systems[key] = Systems(datas[key])
            records[key] = {x["pair_id"]: x for x in datas[key].records}
        rec = records[key][r["pair_id"]]
        view = llm_view(datas[key], rec)
        rows.append(Row(iid, r["pair_id"], r["topic"], r["stratum"], r, predict_full(systems[key], "hybrid_no_gate", rec)[0] == "CONTRADICTS",
                        (rec.get("llm_pair") or {}).get("stance"),
                        (rec["nli_ab"]["contradiction"] + rec["nli_ba"]["contradiction"]) / 2,
                        max(rec["nli_ab"]["entailment"], rec["nli_ba"]["entailment"]), rec["similarity"],
                        systems[key].nli_only(rec)[0],
                        {v: direct_label(rec.get(f"llm_direct_{v}")) for v in ("claims", "context")},
                        old.get(r["pair_id"]), view.factors if view else []))
    return rows


# ------------------------------------------------------------------------------------- four-state systems
def typed_status(factors: list[FactorComparison]) -> str:
    """The simple typed four-state rule: no shared context -> insufficient; a material scope or measurement
    mismatch -> context-dependent; otherwise genuine. What the claims are about (method, baseline) is not context."""
    ctx = [x for x in factors if C.ROLES[x.factor] in C.SUFFICIENCY_ROLES and x.status != "unknown"]
    if not ctx:
        return "insufficient"
    return "context_dependent" if any(x.status == "different" for x in ctx) else "genuine"


def any_mismatch_status(factors: list[FactorComparison]) -> str:
    """As the typed rule, but any stated difference counts, including partial and 'immaterial' ones."""
    ev = [e for e in C.factor_evidence(factors) if e.role in C.SUFFICIENCY_ROLES and e.known]
    if not ev:
        return "insufficient"
    return "context_dependent" if any(e.different > 0 for e in ev) else "genuine"


UP = {"none": "NONE", "genuine": "GENUINE", "context_dependent": "CONTEXT_DEPENDENT", "insufficient": "INSUFFICIENT_EVIDENCE"}
DIRECT = {"CONTRADICTS": "GENUINE", "QUALIFIES": "CONTEXT_DEPENDENT", "INSUFFICIENT": "INSUFFICIENT_EVIDENCE"}


def system_table() -> dict[str, tuple[str, Callable[[Row], str]]]:
    def on_opposed(fn: Callable[[Row], str]) -> Callable[[Row], str]:
        return lambda r: UP[fn(r)] if r.opposed else "NONE"

    table: dict[str, tuple[str, Callable[[Row], str]]] = {
        "all_none": ("Never opposed (constant NONE)", lambda r: "NONE"),
        "nli": ("NLI only (contradiction → GENUINE)", lambda r: "GENUINE" if r.nli_label == "CONTRADICTS" else "NONE"),
        "direct_claims": ("LLM direct, claims only †", lambda r: DIRECT.get(r.direct["claims"] or "", "NONE")),
        "direct_context": ("LLM direct, + passages †", lambda r: DIRECT.get(r.direct["context"] or "", "NONE")),
        "no_context": ("Stance only: every opposed pair GENUINE", on_opposed(lambda r: "genuine")),
        "always_cd": ("Stance + always CONTEXT_DEPENDENT", on_opposed(lambda r: "context_dependent")),
        "always_insufficient": ("Stance + always INSUFFICIENT", on_opposed(lambda r: "insufficient")),
        "suff_then_cd": ("Stance + sufficiency rule, then always CONTEXT_DEPENDENT", on_opposed(lambda r: "insufficient" if r.E == 0 else "context_dependent")),
        "v1": ("V1 hard gate (simple four-state rule)", on_opposed(lambda r: C.v1_status(r.factors))),
        "typed": ("Typed four-state rule (scope/measurement mismatches only)", on_opposed(lambda r: typed_status(r.factors))),
        "any_mismatch": ("Typed rule, any stated difference decides", on_opposed(lambda r: any_mismatch_status(r.factors))),
    }
    for name, q in V2_PARAMS.items():
        table[f"p3_frozen_{name}"] = (f"P3 [{name}], parameters frozen from the forced-label study", on_opposed(lambda r, q=q: C.assess(r.factors, q).status))
    return table


def evaluate4(gold: list[str], pred: list[str]) -> dict:
    out: dict = {"n": len(gold), "accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])) if gold else None}
    f1s = []
    for s in STATES:
        tp = sum(g == s and p == s for g, p in zip(gold, pred))
        n_pred, n_gold = sum(p == s for p in pred), sum(g == s for g in gold)
        prec, rec = (tp / n_pred if n_pred else None), (tp / n_gold if n_gold else None)
        f1 = 2 * tp / (n_pred + n_gold) if (n_pred + n_gold) else None
        out[s] = {"gold": n_gold, "pred": n_pred, "tp": tp, "precision": prec, "recall": rec, "f1": f1}
        if n_gold:
            f1s.append(f1 or 0.0)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
    out["confusion"] = {g: {p: sum(a == g and b == p for a, b in zip(gold, pred)) for p in STATES} for g in STATES}
    return out


def ci(gold: list[str], pred: list[str], metric: Callable[[dict], float | None], rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    g, p = np.array(gold), np.array(pred)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        v = metric(evaluate4(list(g[i]), list(p[i])))
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def pct(v: float | None, lo: float | None = None, hi: float | None = None) -> str:
    if v is None:
        return "–"
    return f"{100 * v:.1f}" + (f" [{100 * lo:.0f}, {100 * hi:.0f}]" if lo is not None and not np.isnan(lo) else "")


def binary_report(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Precision / recall / F1 / accuracy of a binary rule for the positive class, with bootstrap intervals."""
    def stats(i: np.ndarray) -> tuple[float | None, float | None, float | None, float]:
        yy, pp = y[i], pred[i]
        tp = int((yy & pp).sum())
        prec = tp / pp.sum() if pp.sum() else None
        rec = tp / yy.sum() if yy.sum() else None
        f1 = 2 * tp / (pp.sum() + yy.sum()) if (pp.sum() + yy.sum()) else None
        return prec, rec, f1, float((yy == pp).mean())

    base = stats(np.arange(len(y)))
    boots = [stats(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    out = {"n": len(y), "positives": int(y.sum()), "predicted": int(pred.sum()), "tp": int((y & pred).sum())}
    for k, name in enumerate(("precision", "recall", "f1", "accuracy")):
        vals = [b[k] for b in boots if b[k] is not None]
        out[name] = {"value": base[k], "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None}
    return out


def brow(label: str, r: dict) -> list[str]:
    def cell(k: str) -> str:
        v = r[k]
        return "–" if v["value"] is None else pct(v["value"], *(v["ci"] or (None, None)))
    return [label, str(r["n"]), str(r["positives"]), str(r["predicted"]), str(r["tp"]), cell("precision"), cell("recall"), cell("f1"), cell("accuracy")]


# ----------------------------------------------------------------------------------------------- study
def run_study(batch: str = "c4", out_dir: Path | None = None, *, n_boot: int = 2000, n_perm: int = 1000, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = out_dir or EVAL_DIR / "results" / "v2_fourstate"
    rows = load_rows(batch)
    agree = json.loads((EVAL_DIR / "annotation" / batch / "agreement.json").read_text())
    key = json.loads((EVAL_DIR / "annotation" / batch / f"{batch}_key.json").read_text())
    res: dict = {"batch": batch, "pairs": len(rows), "seed": seed}
    md = ["# Four-state development study: opposition, evidence sufficiency, comparability", "",
          f"**Development data, not a paper result.** Batch `{batch}`: {len(rows)} pairs labelled blind by three annotator models "
          "(Gemini 3 Flash, Qwen 3.7 Plus, Claude Haiku 4.5) under the four-state protocol; consensus = majority per dimension; "
          "unadjudicated. The pairs come from the eight topics already used for development.", ""]

    # ---- 1. sample and states
    md += ["## 1. Sample and reference states", ""]
    table = [["Stratum", "Pool", "Annotated", *[SHORT[s] for s in STATES], "no consensus"]]
    for sname, desc in CA.STRATA.items():
        sel = [r for r in rows if r.stratum == sname]
        cnt = Counter(r.status for r in sel)
        table.append([desc, key["design"][sname]["pool"], len(sel), *[cnt[s] for s in STATES], cnt[None]])
    cnt = Counter(r.status for r in rows)
    table.append(["**all**", "", len(rows), *[cnt[s] for s in STATES], cnt[None]])
    md += [md_table(table), "", "Final status as each annotator model labelled it (all pairs):", ""]
    table = [["Annotator", *[SHORT[s] for s in STATES]]]
    for a in ANNOTATOR_IDS:
        c = Counter(r.ref["labels"][a]["status"] for r in rows)
        table.append([a, *[c[s] for s in STATES]])
    md += [md_table(table), ""]
    old_x = Counter((r.old_label, r.status) for r in rows if r.old_label)
    olds = ["CONTRADICTS", "QUALIFIES", "SUPPORTS", "EXTENDS", "UNRELATED"]
    md += ["Forced-choice relation under the first protocol against the four-state status (pairs labelled under both):", "",
           md_table([["First protocol", *[SHORT[s] for s in STATES], "no consensus"],
                     *[[o, *[old_x[(o, s)] for s in STATES], old_x[(o, None)]] for o in olds]]), ""]
    res["states"] = {str(k): v for k, v in cnt.items()}
    res["old_vs_new"] = {f"{o}->{s}": c for (o, s), c in old_x.items()}

    # ---- 2. agreement
    md += ["## 2. Agreement between the annotator models, per dimension", ""]
    table = [["Dimension", "Pairs", "Fleiss κ", "Unanimous", "2–1", "3-way", "Pairwise Cohen κ (gemini–qwen, gemini–haiku, qwen–haiku)"]]
    names = {"stance": "A. face-value stance", "sufficiency": "B. evidence sufficiency (all pairs)",
             "sufficiency_given_majority_oppose": "B. evidence sufficiency (pairs with majority stance oppose)",
             "comparability_all_three_answered": "C. comparability (pairs where all three answered it)",
             "status": "D. final status (all pairs)", "status_given_majority_oppose": "D. final status (pairs with majority stance oppose)"}
    for k, title in names.items():
        a = agree[k]
        table.append([title, a["n"], f(a["fleiss_kappa"]), a["unanimous"], a["majority"], a["split"],
                      ", ".join(f(v["kappa"]) for v in a["pairwise"].values())])
    fa = agree["factors"]
    md += [md_table(table), "",
           f"E. explanatory factors, on the {fa['pairs']} pairs at least two annotators call context-dependent: mean pairwise Jaccard "
           f"overlap of the named factor sets {f(fa['mean_pairwise_jaccard'])}. Named by at least two annotators: "
           + ", ".join(f"{k} {v['named_by_two']}" for k, v in sorted(fa["per_factor"].items(), key=lambda kv: -kv[1]["named_by_two"]) if v["named_by_two"]) + ".", "",
           "Every annotator's final status equals the status its own answers to A–C imply "
           f"({', '.join(str(v['same']) + '/' + str(v['n']) for v in agree['status_matches_derived'].values())}).", ""]
    res["agreement"] = agree

    # ---- 3. stance
    scored = [r for r in rows if r.status]
    md += ["## 3. Opposition: the system's stance against the reference stance", ""]
    table = [["System", "Pairs", "Reference: oppose", "Reference: not oppose"]]
    for label, sel in (("reads the pair as opposed", [r for r in rows if r.opposed]), ("does not", [r for r in rows if not r.opposed])):
        table.append([label, len(sel), sum(r.ref["stance"] == "oppose" for r in sel), sum(r.ref["stance"] != "oppose" for r in sel)])
    so = [r for r in rows if r.opposed]
    md += [md_table(table), "",
           f"Of the {len(so)} pairs the system reads as opposed, the annotator majority agrees for {sum(r.ref['stance'] == 'oppose' for r in so)} "
           f"({100 * np.mean([r.ref['stance'] == 'oppose' for r in so]):.0f}%). The non-opposed pairs were sampled, not exhaustive, so the "
           "second row is not a recall estimate.", ""]
    res["stance"] = {"system_opposed": len(so), "of_which_reference_oppose": sum(r.ref["stance"] == "oppose" for r in so),
                     "not_opposed_annotated": len(rows) - len(so), "of_which_reference_oppose_missed": sum(r.ref["stance"] == "oppose" for r in rows if not r.opposed)}

    # ---- 4. Table A: sufficiency
    ref_opp = [r for r in scored if r.ref["stance"] == "oppose" and r.status != "NONE"]
    both = [r for r in ref_opp if r.opposed]
    md += ["## 4. TABLE A — evidence sufficiency", "",
           "System rule (fixed in advance, nothing fitted): *insufficient* when no scope or measurement factor is stated by both "
           "papers (E = 0). Positive class: INSUFFICIENT.", ""]
    table = [["Population and reference", "Pairs", "Reference insufficient", "System says insufficient", "Both", "Precision", "Recall", "F1", "Accuracy"]]
    A: dict = {}
    pops: list[tuple[str, list[Row], Callable[[Row], bool]]] = [
        ("Opposed by reference and by system; final status INSUFFICIENT vs GENUINE/CONTEXT_DEPENDENT", both, lambda r: r.status == "INSUFFICIENT_EVIDENCE"),
        ("Opposed by reference (any system stance); same target", ref_opp, lambda r: r.status == "INSUFFICIENT_EVIDENCE"),
        ("All annotated pairs; dimension B alone (sufficiency majority)", rows, lambda r: r.ref["sufficiency"] == "insufficient"),
    ] + [(f"Annotator {a} alone: pairs it calls opposed, its own sufficiency answer", [r for r in rows if r.ref["labels"][a]["stance"] == "oppose" and r.opposed],
          (lambda r, a=a: r.ref["labels"][a]["sufficiency"] == "insufficient")) for a in ANNOTATOR_IDS]
    for label, sel, target in pops:
        y = np.array([target(r) for r in sel])
        pred = np.array([r.E == 0 for r in sel])
        rep = binary_report(y, pred, rng, n_boot)
        e = np.array([r.E for r in sel])
        nk = np.array([sum(ev.known for ev in S._ev(r)) for r in sel])  # type: ignore[arg-type]
        rep["auc_E"] = S.auc(y.astype(int), -e)
        rep["auc_E_ci"] = S.boot_ci(lambda i: S.auc(y[i].astype(int), -e[i]), len(sel), rng, n_boot)
        rep["auc_known_any_role"] = S.auc(y.astype(int), -nk)
        tp, fp = int((y & pred).sum()), int((~y & pred).sum())
        rep["fisher_p"] = fisher_exact_two_sided(tp, fp, int((y & ~pred).sum()), int((~y & ~pred).sum()))
        A[label] = rep
        table.append(brow(label, rep))
    md += [md_table(table), "", "Ranking by the amount of shared context instead of the E = 0 cut (AUC; 0.50 = no signal):", ""]
    table = [["Population", "AUC, fewer shared context factors → insufficient [95% CI]", "AUC, fewer shared factors of any role", "Fisher p for the E = 0 rule"]]
    for label, rep in A.items():
        table.append([label, f"{f(rep['auc_E'])} [{f(rep['auc_E_ci'][0])}, {f(rep['auc_E_ci'][1])}]", f(rep["auc_known_any_role"]), f"{rep['fisher_p']:.3f}"])
    md += [md_table(table), ""]
    table = [["Shared context E (system) on pairs opposed by reference and system", "Pairs", *[SHORT[s] for s in STATES[1:]]]]
    for lo, hi, label in ((0, 0.01, "0"), (0.01, 1.01, "0.5–1"), (1.01, 2.01, "1.5–2"), (2.01, 99, "> 2")):
        sel = [r for r in both if lo <= r.E < hi]
        if sel:
            c = Counter(r.status for r in sel)
            table.append([label, len(sel), *[c[s] for s in STATES[1:]]])
    md += [md_table(table), ""]
    res["table_a"] = A

    # ---- 5. Table B: comparability
    B_pop = [r for r in both if r.status in ("GENUINE", "CONTEXT_DEPENDENT")]
    yb = np.array([r.status == "GENUINE" for r in B_pop])
    md += ["## 5. TABLE B — context comparability (the P3 hypothesis)", "",
           f"Population: pairs opposed by reference and by system whose reference status is GENUINE or CONTEXT_DEPENDENT "
           f"(sufficient evidence): **{len(B_pop)} pairs, {int(yb.sum())} GENUINE, {int((~yb).sum())} CONTEXT_DEPENDENT.** "
           "Insufficient pairs are excluded. Positive class: GENUINE.", ""]
    table = [["Rule (nothing fitted)", "Pairs", "Reference GENUINE", "System says GENUINE", "Both", "Precision", "Recall", "F1", "Accuracy"]]
    Bres: dict = {"n": len(B_pop), "genuine": int(yb.sum())}
    rules: list[tuple[str, Callable[[Row], bool]]] = [
        ("Always CONTEXT_DEPENDENT", lambda r: False),
        ("V1 hard gate: no material mismatch in any factor", lambda r: C.v1_status(r.factors) == "genuine"),
        ("Typed rule: no material scope/measurement mismatch", lambda r: typed_status(r.factors) == "genuine"),
        ("Typed rule, any stated difference decides", lambda r: any_mismatch_status(r.factors) == "genuine"),
        ("No shared context at all (E = 0)", lambda r: r.E == 0),
    ]
    for label, fn in rules:
        rep = binary_report(yb, np.array([fn(r) for r in B_pop]), rng, n_boot)
        Bres[label] = rep
        table.append(brow(label, rep))
    md += [md_table(table), ""]
    tb = np.array([r.topic for r in B_pop])
    folds = S.topic_folds(B_pop)  # type: ignore[arg-type]
    md += ["Graded models, leave-one-topic-out. AUC compares pairs within a topic; log-loss skill is relative to the constant model.", ""]
    table = [["Model", "Fitted numbers", "AUC [95% CI]", "Log-loss", "Skill vs constant"]]
    cvb: dict = {}
    model_names = ["constant", "context_stated", "uniform_ctx", "uniform", "hand", "role", "role4", "free12"]
    base_loss = None
    for name in model_names:
        spec = S.SPEC[name]
        oof, _ = S.cv_scores(B_pop, spec, yb.astype(float), folds)  # type: ignore[arg-type]
        loss = S.log_loss(yb.astype(int), oof)
        base_loss = loss if name == "constant" else base_loss
        a = S.auc_within(yb.astype(int), oof, tb) if spec.columns else 0.5
        lo, hi = S.boot_ci(lambda i: S.auc_within(yb[i].astype(int), oof[i], tb[i]), len(B_pop), rng, n_boot) if spec.columns else (0.5, 0.5)
        cvb[name] = {"auc": a, "ci": [lo, hi], "log_loss": loss, "skill": 1 - loss / base_loss, "n_params": spec.n_params}
        table.append([spec.title, spec.n_params, f"{f(a)} [{f(lo)}, {f(hi)}]" if spec.columns else "0.50", f(loss, 3), f"{100 * (1 - loss / base_loss):+.1f}%"])
    md += [md_table(table), ""]
    # shuffled-factor control on Table B
    perm = {}
    for name in ("uniform_ctx", "role"):
        spec = S.SPEC[name]
        vals = []
        for _ in range(n_perm):
            Xs = S.design(S.shuffled_rows(B_pop, rng), spec)  # type: ignore[arg-type]
            o = np.zeros(len(B_pop))
            for test in folds:
                train = np.setdiff1d(np.arange(len(B_pop)), test)
                o[test] = S.predict_proba(Xs[test], S.fit_logistic(Xs[train], yb[train].astype(float), spec.nonneg, iters=600))
            v = S.auc_within(yb.astype(int), o, tb)
            if v is not None:
                vals.append(v)
        va = np.array(vals)
        perm[name] = {"real": cvb[name]["auc"], "mean": float(va.mean()), "lo": float(np.percentile(va, 2.5)), "hi": float(np.percentile(va, 97.5)),
                      "p": float((1 + (va >= cvb[name]["auc"] - 1e-12).sum()) / (1 + len(va)))}
    md += [f"Shuffled-factor control ({n_perm} shuffles within topic): " + "; ".join(
        f"{S.SPEC[n].title}: real AUC {f(p['real'])}, shuffled {f(p['mean'])} [{f(p['lo'])}, {f(p['hi'])}], p = {f(p['p'], 3)}" for n, p in perm.items()) + ".", ""]
    # the same question against softer targets, because consensus GENUINE is rare
    md += ["Because consensus GENUINE is rare, the same question against softer targets on pairs opposed by both the annotator "
           "majority and the system (exploratory). AUC within topic of each score for GENUINE; the first, second and last "
           "three columns need no fitting, the role-weight column is fitted leave-one-topic-out.", ""]
    soft_pop = [r for r in rows if r.ref["stance"] == "oppose" and r.opposed]
    cols = [("ctx", "Uniform: fewer context mismatches"), ("hand", "V1 hand weights"), ("role", "P3 role weights (fitted)"),
            ("v1", "V1 gate (yes/no)"), ("typed", "Typed rule (yes/no)"), ("e0", "No shared context (E = 0)"),
            ("det", "Uniform, named values compared in code")]
    table = [["Target", "Pairs", "GENUINE", *[c[1] for c in cols]]]
    soft = {}
    targets: list[tuple[str, Callable[[Row], bool], Callable[[Row], bool]]] = [
        ("consensus status (Table B population)", lambda r: r.status in ("GENUINE", "CONTEXT_DEPENDENT"), lambda r: r.status == "GENUINE"),
        ("at least one annotator says GENUINE (vs none)", lambda r: True, lambda r: any(l["status"] == "GENUINE" for l in r.ref["labels"].values())),
    ] + [(f"annotator {a} alone: GENUINE vs CONTEXT_DEPENDENT", (lambda r, a=a: r.ref["labels"][a]["status"] in ("GENUINE", "CONTEXT_DEPENDENT")),
          (lambda r, a=a: r.ref["labels"][a]["status"] == "GENUINE")) for a in ANNOTATOR_IDS]
    for label, keep, pos in targets:
        sel = [r for r in soft_pop if keep(r)]
        y = np.array([int(pos(r)) for r in sel])
        tp = np.array([r.topic for r in sel])
        det = [dataclasses.replace(r, factors=C.with_deterministic_statuses(r.factors)) for r in sel]
        role_oof, _ = S.cv_scores(sel, S.SPEC["role"], y.astype(float), S.topic_folds(sel))  # type: ignore[arg-type]
        sc = {"ctx": -np.array([S._diff(r, roles=S.CONTEXT_ROLES) for r in sel]),  # type: ignore[arg-type]
              "hand": -np.array([S._diff(r, weights=S.HAND_WEIGHTS) for r in sel]),  # type: ignore[arg-type]
              "role": role_oof,
              "v1": np.array([float(C.v1_status(r.factors) == "genuine") for r in sel]),
              "typed": np.array([float(typed_status(r.factors) == "genuine") for r in sel]),
              "e0": np.array([float(r.E == 0) for r in sel]),
              "det": -np.array([S._diff(r, roles=S.CONTEXT_ROLES) for r in det])}  # type: ignore[arg-type]
        soft[label] = {"n": len(sel), "positives": int(y.sum())}
        cells = []
        for k, _ in cols:
            a = S.auc_within(y, sc[k], tp)
            lo, hi = S.boot_ci(lambda i, v=sc[k]: S.auc_within(y[i], v[i], tp[i]), len(sel), rng, min(n_boot, 1000))
            soft[label][k] = {"auc": a, "ci": [lo, hi]}
            cells.append(f"{f(a)} [{f(lo)}, {f(hi)}]")
        # is the graded score better than the yes/no rule? paired bootstrap of the AUC difference
        d = [x for x in (((S.auc_within(y[i], sc["hand"][i], tp[i]) or np.nan) - (S.auc_within(y[i], sc["v1"][i], tp[i]) or np.nan))
                         for i in (rng.integers(0, len(sel), len(sel)) for _ in range(min(n_boot, 1000)))) if not np.isnan(x)]
        soft[label]["hand_minus_v1"] = {"delta": (soft[label]["hand"]["auc"] or 0) - (soft[label]["v1"]["auc"] or 0),
                                        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float(np.mean(np.array(d) <= 0))}
        table.append([label, len(sel), int(y.sum()), *cells])
    md += [md_table(table), "", "Graded score against the yes/no gate (AUC, V1 hand weights minus V1 gate; paired bootstrap):", "",
           md_table([["Target", "Δ AUC [95% CI]", "share of resamples with Δ ≤ 0"],
                     *[[k, f"{v['hand_minus_v1']['delta']:+.2f} [{v['hand_minus_v1']['ci'][0]:+.2f}, {v['hand_minus_v1']['ci'][1]:+.2f}]",
                        f(v["hand_minus_v1"]["p_le_0"], 3)] for k, v in soft.items()]]), ""]
    # deterministic comparison of stated values
    x = Counter()
    for r in so:
        for fc in r.factors:
            det_status = C.deterministic_status(fc.factor, fc.a, fc.b) if fc.status != "unknown" else None
            if det_status:
                llm_status = "different (immaterial)" if fc.note == C.IMMATERIAL_NOTE else fc.status
                x[(fc.factor, llm_status, det_status)] += 1
    md += ["Deterministic comparison of stated values (names as sets, quantities as numbers) against the status the LLM gave, "
           "over the factor entries of system-opposed pairs where code can compare:", "",
           md_table([["Factor", "LLM status", "code: same", "code: overlap", "code: different"],
                     *[[fac, st, x[(fac, st, "same")], x[(fac, st, "overlap")], x[(fac, st, "different")]]
                       for fac in C.FACTOR_ORDER for st in ("same", "overlap", "different (immaterial)", "different")
                       if any(x[(fac, st, d)] for d in ("same", "overlap", "different"))]]), "",
           "Free-text factors (experimental conditions, training & design, population, task, methodology) cannot be compared in "
           "code and keep the LLM's status.", ""]
    det_agree = {f"{a}|{b}|{c}": n for (a, b, c), n in x.items()}
    res["table_b"] = {"rules": Bres, "cv": cvb, "shuffled": perm, "soft_targets": soft, "deterministic_vs_llm": det_agree}

    # ---- 6. end to end
    md += ["## 6. End-to-end four-state decision", "",
           f"All {len(scored)} annotated pairs with a consensus status ({', '.join(f'{SHORT[s]} {sum(r.status == s for r in scored)}' for s in STATES)}). "
           "The sample over-represents opposed pairs, so these are not prevalence-weighted figures. Macro-F1 averages the four "
           "states; with so few GENUINE pairs it is unstable. † a direct-LLM QUALIFIES is read as CONTEXT_DEPENDENT although "
           "that prompt did not require face-value opposition.", ""]
    systems = system_table()
    gold = [r.status for r in scored]
    table = [["System", "Accuracy [95% CI]", "Macro-F1 [95% CI]", "GENUINE: asserted / correct", "CONTEXT-DEP. F1", "INSUFFICIENT F1", "NONE F1"]]
    e2e: dict = {}
    for name, (title, fn) in systems.items():
        pred = [fn(r) for r in scored]
        ev = evaluate4(gold, pred)  # type: ignore[arg-type]
        ev["accuracy_ci"] = ci(gold, pred, lambda e: e["accuracy"], rng, min(n_boot, 1000))  # type: ignore[arg-type]
        ev["macro_f1_ci"] = ci(gold, pred, lambda e: e["macro_f1"], rng, min(n_boot, 1000))  # type: ignore[arg-type]
        e2e[name] = {"title": title, **ev}
        table.append([title, pct(ev["accuracy"], *ev["accuracy_ci"]), pct(ev["macro_f1"], *ev["macro_f1_ci"]),
                      f"{ev['GENUINE']['pred']} / {ev['GENUINE']['tp']}", pct(ev["CONTEXT_DEPENDENT"]["f1"]), pct(ev["INSUFFICIENT_EVIDENCE"]["f1"]), pct(ev["NONE"]["f1"])])
    md += [md_table(table), ""]
    dl = [r for r in scored if r.opposed and r.ref["stance"] == "oppose" and r.status != "NONE"]
    gold_dl = [r.status for r in dl]
    md += [f"Decision layer alone: the {len(dl)} pairs opposed by both reference and system "
           f"({', '.join(f'{SHORT[s]} {sum(g == s for g in gold_dl)}' for s in STATES[1:])}), three states.", ""]
    table = [["Decision rule", "Accuracy [95% CI]", "GENUINE P / R", "CONTEXT-DEP. P / R", "INSUFFICIENT P / R"]]
    dl_res = {}
    for name in ("no_context", "always_cd", "always_insufficient", "suff_then_cd", "v1", "typed", "any_mismatch", *[n for n in systems if n.startswith("p3_frozen")]):
        title, fn = systems[name]
        pred = [fn(r) for r in dl]
        ev = evaluate4(gold_dl, pred)  # type: ignore[arg-type]
        lo, hi = ci(gold_dl, pred, lambda e: e["accuracy"], rng, min(n_boot, 1000))  # type: ignore[arg-type]
        dl_res[name] = {"title": title, **ev, "accuracy_ci": [lo, hi]}
        table.append([title.replace("Stance + ", "").replace("Stance only: ", ""), pct(ev["accuracy"], lo, hi),
                      *[f"{pct(ev[s]['precision'])} / {pct(ev[s]['recall'])}" for s in STATES[1:]]])
    md += [md_table(table), "", "Confusion of the V1 hard gate on all scored pairs (rows: reference, columns: system):", "",
           md_table([["", *[SHORT[s] for s in STATES]], *[[SHORT[g], *[e2e["v1"]["confusion"][g][p] for p in STATES]] for g in STATES]]), ""]
    res["end_to_end"], res["decision_layer"] = e2e, dl_res

    # ---- 7. any shared context, retested
    md += ["## 7. The 'any shared context' finding, retested", "",
           "Under the forced-choice labels, opposed pairs with no shared context were 'contradictions' 10/16 times against 9/52. "
           "The same split against the four-state labels (pairs the system reads as opposed, with a consensus status):", ""]
    so_scored = [r for r in scored if r.opposed]
    table = [["Shared context (system)", "Pairs", *[SHORT[s] for s in STATES], "old protocol: CONTRADICTS / labelled"]]
    retest = {}
    for label, sel in (("E = 0", [r for r in so_scored if r.E == 0]), ("E > 0", [r for r in so_scored if r.E > 0])):
        c = Counter(r.status for r in sel)
        old_l = [r for r in sel if r.old_label]
        retest[label] = {"n": len(sel), **{s: c[s] for s in STATES}, "old_contradicts": sum(r.old_label == "CONTRADICTS" for r in old_l), "old_n": len(old_l)}
        table.append([label, len(sel), *[c[s] for s in STATES], f"{retest[label]['old_contradicts']} / {len(old_l)}"])
    z, k = retest["E = 0"], retest["E > 0"]
    p_gen = fisher_exact_two_sided(z["GENUINE"], z["n"] - z["GENUINE"], k["GENUINE"], k["n"] - k["GENUINE"])
    p_ins = fisher_exact_two_sided(z["INSUFFICIENT_EVIDENCE"], z["n"] - z["INSUFFICIENT_EVIDENCE"], k["INSUFFICIENT_EVIDENCE"], k["n"] - k["INSUFFICIENT_EVIDENCE"])
    md += [md_table(table), "", f"Fisher's exact test, E = 0 versus E > 0: GENUINE p = {p_gen:.3f}; INSUFFICIENT p = {p_ins:.3f}.", ""]
    retest["fisher_genuine"], retest["fisher_insufficient"] = p_gen, p_ins
    res["shared_context_retest"] = retest

    # ---- 8. are the factors explanatory?
    md += ["## 8. Is the factor table explanatory? System mismatches against the factors annotators name", "",
           "Pairs opposed by the system whose reference status is CONTEXT_DEPENDENT. Reference set: factors named by at least two "
           "annotators (pairs where no factor reaches two votes are scored against the union and reported separately). "
           "System set: factors with a material mismatch. Chance: the same system sets reassigned among these pairs within a topic.", ""]
    cd = [r for r in scored if r.opposed and r.status == "CONTEXT_DEPENDENT"]

    def match(sys_sets: list[set[str]], ref_sets: list[set[str]]) -> dict:
        tp = sum(len(s & g) for s, g in zip(sys_sets, ref_sets))
        ns, ng = sum(map(len, sys_sets)), sum(map(len, ref_sets))
        return {"precision": tp / ns if ns else None, "recall": tp / ng if ng else None,
                "jaccard": float(np.mean([len(s & g) / len(s | g) if s | g else 1.0 for s, g in zip(sys_sets, ref_sets)])),
                "any_hit": float(np.mean([bool(s & g) for s, g in zip(sys_sets, ref_sets)]))}

    expl = {}
    table = [["Reference set", "System set", "Pairs", "Precision", "Recall", "Mean Jaccard", "Pairs with ≥ 1 shared factor", "Chance: ≥ 1 shared [95%]", "p"]]
    ref_variants = [("named by ≥ 2 annotators", [r for r in cd if r.ref["factors"]], lambda r: set(r.ref["factors"])),
                    ("named by any annotator", [r for r in cd if r.ref["factor_votes"]], lambda r: set(r.ref["factor_votes"]))]
    sys_variants: list[tuple[str, Callable[[Row], set[str]]]] = [
        ("material mismatch (status different)", lambda r: {x.factor for x in r.factors if x.status == "different"}),
        ("any stated difference", lambda r: {e.factor for e in C.factor_evidence(r.factors) if e.different > 0}),
        ("always 'experimental conditions' (most frequent)", lambda r: {"experimental conditions"}),
    ]
    for rlabel, sel, rfn in ref_variants:
        refs = [rfn(r) for r in sel]
        tps = np.array([r.topic for r in sel])
        for slabel, sfn in sys_variants:
            sys_sets = [sfn(r) for r in sel]
            m = match(sys_sets, refs)
            null = []
            if "always" not in slabel:
                for _ in range(n_perm):
                    order = np.arange(len(sel))
                    for t in set(tps):
                        idx = np.flatnonzero(tps == t)
                        order[idx] = rng.permutation(idx)
                    null.append(match([sys_sets[j] for j in order], refs)["any_hit"])
            pv = float((1 + (np.array(null) >= m["any_hit"] - 1e-12).sum()) / (1 + len(null))) if null else None
            expl[f"{rlabel} | {slabel}"] = {**m, "n": len(sel), "null_mean": float(np.mean(null)) if null else None, "p": pv}
            table.append([rlabel, slabel, len(sel), pct(m["precision"]), pct(m["recall"]), f(m["jaccard"]), pct(m["any_hit"]),
                          f"{pct(float(np.mean(null)))} [{100 * np.percentile(null, 2.5):.0f}, {100 * np.percentile(null, 97.5):.0f}]" if null else "–",
                          f(pv, 3) if pv is not None else "–"])
    md += [md_table(table), "", f"For scale: two annotator models name factor sets with mean Jaccard overlap {f(fa['mean_pairwise_jaccard'])}.", ""]
    table = [["Factor", "Role", "System: material mismatch", "Annotators: named by ≥ 2", "Both", "Named by any annotator"]]
    per_factor = {}
    for name in C.FACTOR_ORDER:
        s_has = [any(x.factor == name and x.status == "different" for x in r.factors) for r in cd]
        g_has = [name in r.ref["factors"] for r in cd]
        any_has = [name in r.ref["factor_votes"] for r in cd]
        per_factor[name] = {"system": sum(s_has), "reference": sum(g_has), "both": sum(a and b for a, b in zip(s_has, g_has)), "any": sum(any_has)}
        table.append([name, C.ROLES[name], sum(s_has), sum(g_has), per_factor[name]["both"], sum(any_has)])
    md += [f"Per factor, over the {len(cd)} context-dependent pairs:", "", md_table(table), ""]
    res["explanatory"] = {"n_context_dependent": len(cd), "match": expl, "per_factor": per_factor}

    # ---- 9. ablations of the sufficiency count
    md += ["## 9. Factor and role ablations", "",
           "Table A is the only place the factor evidence shows a measurable signal, so the ablation is run there: AUC of "
           "'less shared context → insufficient' on pairs opposed by reference and system, with one factor or role left out of the count.", ""]
    ya = np.array([int(r.status == "INSUFFICIENT_EVIDENCE") for r in both])

    def e_without(r: Row, pred: Callable[[C.FactorEvidence], bool]) -> float:
        return sum(ev.grounding for ev in S._ev(r) if ev.known and not pred(ev))  # type: ignore[arg-type]

    base = S.auc(ya, -np.array([e_without(r, lambda ev: ev.role not in C.SUFFICIENCY_ROLES) for r in both]))
    table = [["Count of shared factors", "AUC", "Δ vs scope + measurement"]]
    abl = {"scope + measurement (the rule)": base}
    variants: list[tuple[str, Callable[[C.FactorEvidence], bool]]] = [
        ("all twelve factors", lambda ev: False), ("scope only", lambda ev: ev.role != "scope"), ("measurement only", lambda ev: ev.role != "measurement"),
        ("intervention only (method, baseline)", lambda ev: ev.role != "intervention"),
    ] + [(f"scope + measurement without {n}", (lambda ev, n=n: ev.role not in C.SUFFICIENCY_ROLES or ev.factor == n))
         for n in C.FACTOR_ORDER if C.ROLES[n] in C.SUFFICIENCY_ROLES and any(ev.known for r in both for ev in S._ev(r) if ev.factor == n)]  # type: ignore[arg-type]
    table.append(["scope + measurement (the rule)", f(base), "–"])
    for label, pred in variants:
        a = S.auc(ya, -np.array([e_without(r, pred) for r in both]))
        abl[label] = a
        table.append([label, f(a), f"{a - base:+.2f}"])
    w = np.array([e_without(r, lambda ev: False) for r in both])
    abl["grounding weight off (paper-level statements count fully)"] = S.auc(ya, -np.array(
        [sum(1.0 for ev in S._ev(r) if ev.known and ev.role in C.SUFFICIENCY_ROLES) for r in both]))  # type: ignore[arg-type]
    table.append(["scope + measurement, paper-level statements count fully", f(abl["grounding weight off (paper-level statements count fully)"]),
                  f"{abl['grounding weight off (paper-level statements count fully)'] - base:+.2f}"])
    never = [n for n in C.FACTOR_ORDER if not any(ev.known for r in so for ev in S._ev(r) if ev.factor == n)]  # type: ignore[arg-type]
    md += [md_table(table), "", f"Factors never stated by both papers in any system-opposed pair: {', '.join(never) or 'none'}.", ""]
    res["ablations"] = abl

    # ---- summary
    a1 = A["Opposed by reference and by system; final status INSUFFICIENT vs GENUINE/CONTEXT_DEPENDENT"]
    v1b = Bres["V1 hard gate: no material mismatch in any factor"]
    key_match = expl["named by ≥ 2 annotators | material mismatch (status different)"]
    summary = ["## Summary", "",
               f"1. **States.** Of {len(rows)} pairs: {', '.join(f'{SHORT[s]} {cnt[s]}' for s in STATES)}, no consensus {cnt[None]}. "
               f"Consensus GENUINE contradictions are rare ({cnt['GENUINE']}); the annotators differ widely on that state "
               f"({', '.join(a + ' ' + str(sum(r.ref['labels'][a]['status'] == 'GENUINE' for r in rows)) for a in ANNOTATOR_IDS)}). "
               f"Of the 25 forced-choice 'contradictions' of the first protocol, {old_x[('CONTRADICTS', 'CONTEXT_DEPENDENT')]} are now "
               f"context-dependent, {old_x[('CONTRADICTS', 'INSUFFICIENT_EVIDENCE')]} insufficient, {old_x[('CONTRADICTS', 'GENUINE')]} genuine.",
               f"2. **Agreement is moderate on stance and weak below it**: Fleiss κ {f(agree['stance']['fleiss_kappa'])} for stance, "
               f"{f(agree['sufficiency_given_majority_oppose']['fleiss_kappa'])} for sufficiency and "
               f"{f(agree['status_given_majority_oppose']['fleiss_kappa'])} for the final status on opposed pairs.",
               f"3. **Opposition is the weakest step of the system.** The annotator majority sees opposition in only "
               f"{res['stance']['of_which_reference_oppose']} of the {len(so)} pairs the system reads as opposed.",
               f"4. **Table A (sufficiency).** The E = 0 rule finds {a1['tp']} of {a1['positives']} INSUFFICIENT pairs with "
               f"{a1['predicted']} predictions (precision {pct(a1['precision']['value'])}, recall {pct(a1['recall']['value'])}; "
               f"AUC {f(a1['auc_E'])} [{f(a1['auc_E_ci'][0])}, {f(a1['auc_E_ci'][1])}]).",
               f"5. **Table B (comparability, the P3 hypothesis) is underpowered**: {int(yb.sum())} GENUINE against "
               f"{int((~yb).sum())} CONTEXT_DEPENDENT pairs. Yes/no rules do badly (the V1 gate calls {v1b['predicted']} genuine, {v1b['tp']} "
               f"correctly). Graded scores have point estimates above chance but intervals that reach it: uniform AUC {f(cvb['uniform_ctx']['auc'])} "
               f"[{f(cvb['uniform_ctx']['ci'][0])}, {f(cvb['uniform_ctx']['ci'][1])}], V1 hand weights {f(cvb['hand']['auc'])} "
               f"[{f(cvb['hand']['ci'][0])}, {f(cvb['hand']['ci'][1])}], fitted role weights {f(cvb['role']['auc'])} "
               f"[{f(cvb['role']['ci'][0])}, {f(cvb['role']['ci'][1])}]; shuffled-factor p = {f(perm['uniform_ctx']['p'], 3)} (uniform). "
               f"Against 'at least one annotator says GENUINE' ({soft['at least one annotator says GENUINE (vs none)']['positives']} positives) "
               f"hand weights give {f(soft['at least one annotator says GENUINE (vs none)']['hand']['auc'])} "
               f"[{f(soft['at least one annotator says GENUINE (vs none)']['hand']['ci'][0])}, {f(soft['at least one annotator says GENUINE (vs none)']['hand']['ci'][1])}]. "
               "Fitted role weights never beat uniform or hand weights.",
               f"6. **'Any shared context'** no longer predicts contradictions: GENUINE {z['GENUINE']}/{z['n']} with E = 0 against "
               f"{k['GENUINE']}/{k['n']} with E > 0 (p = {p_gen:.2f}). Nor does it mark the pairs annotators call INSUFFICIENT: "
               f"{z['INSUFFICIENT_EVIDENCE']}/{z['n']} against {k['INSUFFICIENT_EVIDENCE']}/{k['n']} (p = {p_ins:.2f}). The earlier finding "
               "was an artefact of the forced-choice labels.",
               f"7. **Explanatory value.** On {key_match['n']} context-dependent pairs the system's material mismatches share at least one "
               f"factor with the annotators' in {pct(key_match['any_hit'])}% of pairs (chance {pct(key_match['null_mean'])}%, p = {f(key_match['p'], 3)}); "
               f"precision {pct(key_match['precision'])}%, recall {pct(key_match['recall'])}%.", ""]
    md[4:4] = summary
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fourstate_study.md").write_text("\n".join(md))
    (out_dir / "fourstate_study.json").write_text(json.dumps(res, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    return res
