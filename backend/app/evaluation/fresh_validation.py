"""Fresh-topic validation of the frozen E1 opposition detectors (batch f1).

Written and frozen before any fresh pair was labelled; the design, estimators, criteria and verdict rule are in
``eval/annotation/f1/PREREGISTRATION.md`` and are implemented here as written. Nothing in this module is fitted,
searched or chosen from the fresh labels. No LLM is called here: the readings come from the frozen pipeline
(``cg_eval.py prepare`` / ``llm``) and the labels from the C4 annotation protocol (``fourstate-annotate``).

In every system below the LLM only reads the pair; a fixed rule decides whether opposition is established and the
existing V1 gate decides the conflict status.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from app.evaluation import EVAL_DIR
from app.evaluation import fourstate_study as FS
from app.evaluation.experiment import Systems, TopicData, direct_label, llm_view, load_topic, predict_full
from app.evaluation.layer_study import Boot, md_table, pct
from app.reasoning import comparability as C

RUN = "fresh"
SEED = 20261007
COST_PER_PAIR = 0.0040  # US$, three annotators; measured on batch c4 (0.758 / 196, retries included)
RESERVE = 0.015
STRATA = {
    "A": "pair-analysis stance = oppose and relation call with passages = CONTRADICTS (the E1 both-agree set)",
    "B": "the current trigger fires, not in A",
    "C": "the current trigger does not fire; both relation calls say CONTRADICTS",
    "D": "the current trigger does not fire; exactly one relation call says CONTRADICTS",
    "E": "no LLM reading signals opposition",
}
DETECTORS = {
    "current": "Current trigger (stance oppose, or partial + NLI)",
    "rel_ctx": "Relation call with passages = CONTRADICTS",
    "agree": "E1 both agree: stance oppose AND relation call",
    "vote2": "E1 2 of 3: stance, relation with passages, relation claims only",
    "nli": "NLI only (frozen comparison)",
}
E1 = ("agree", "vote2")


def readings(data: TopicData) -> list[dict]:
    """The frozen readings and detector outputs of every candidate pair of a topic."""
    systems = Systems(data)
    out = []
    for rec in data.records:
        stance = (rec.get("llm_pair") or {}).get("stance") == "oppose"
        rel_ctx = direct_label(rec.get("llm_direct_context")) == "CONTRADICTS"
        rel_claims = direct_label(rec.get("llm_direct_claims")) == "CONTRADICTS"
        trigger = predict_full(systems, "hybrid_no_gate", rec)[0] == "CONTRADICTS"
        view = llm_view(data, rec)
        a, b = data.claims[rec["a"]], data.claims[rec["b"]]
        row = {"pair_id": rec["pair_id"], "topic": data.topic.id, "cluster": "|".join(sorted((a.paper_id, b.paper_id))),
               "complete": all(rec.get(k) for k in ("llm_pair", "llm_direct_context", "llm_direct_claims")),
               "current": trigger, "rel_ctx": rel_ctx, "rel_claims": rel_claims, "stance_oppose": stance,
               "agree": stance and rel_ctx, "vote2": (int(stance) + int(rel_ctx) + int(rel_claims)) >= 2,
               "nli": systems.nli_only(rec)[0] == "CONTRADICTS", "v1": C.v1_status(view.factors if view else [])}
        row["stratum"] = ("A" if row["agree"] else "B" if trigger else "C" if (rel_ctx and rel_claims) else "D" if (rel_ctx or rel_claims) else "E")
        out.append(row)
    return out


def allocate(sizes: dict[str, int], n_max: int) -> dict[str, int]:
    """How many pairs of each stratum to label. A, then B, up to 75% of the budget; the rest is split 40 / 35 / 25
    over C, D, E, and what a small stratum cannot use passes to the next one."""
    n = {k: 0 for k in STRATA}
    head = int(0.75 * n_max)
    n["A"] = min(sizes["A"], head)
    n["B"] = min(sizes["B"], head - n["A"])
    rest = n_max - n["A"] - n["B"]
    quota = {"C": int(np.ceil(0.40 * rest)), "D": int(np.ceil(0.35 * rest))}
    n["C"] = min(sizes["C"], quota["C"])
    n["D"] = min(sizes["D"], quota["D"] + (quota["C"] - n["C"]), rest - n["C"])
    n["E"] = min(sizes["E"], rest - n["C"] - n["D"])
    return n


def sample(datas: dict[tuple[str, str], TopicData], n_max: int, batch: str, seed: int = SEED) -> tuple[list[dict], dict]:
    """Stratified random sample, pooled over the fresh topics. System outputs choose pairs; they never reach a prompt."""
    rows = [r for _, data in sorted(datas.items()) for r in readings(data)]
    by = {k: sorted(r["pair_id"] for r in rows if r["stratum"] == k) for k in STRATA}
    take = allocate({k: len(v) for k, v in by.items()}, n_max)
    rng = random.Random(seed)
    chosen: dict[str, str] = {}
    for k in STRATA:
        ids = list(by[k])
        rng.shuffle(ids)
        chosen.update({pid: k for pid in ids[: take[k]]})
    topic_of = {r["pair_id"]: r["topic"] for r in rows}
    items = [{"item_id": f"{batch}_" + hashlib.sha1(f"{seed}:{pid}".encode()).hexdigest()[:10], "pair_id": pid, "topic": topic_of[pid],
              "run": RUN, "stratum": k} for pid, k in sorted(chosen.items())]
    design = {k: {"pool": len(by[k]), "sampled": take[k], "weight": (len(by[k]) / take[k] if take[k] else None)} for k in STRATA}
    pool = {t: dict(Counter(r["stratum"] for r in rows if r["topic"] == t)) for t in sorted({r["topic"] for r in rows})}
    return items, {"strata": design, "pool_by_topic": pool, "candidates": len(rows), "incomplete_readings": sum(not r["complete"] for r in rows),
                   "detector_pool_counts": {d: sum(r[d] for r in rows) for d in DETECTORS}}


# ------------------------------------------------------------------------------------------------ scoring
def load(batch: str) -> tuple[list[dict], dict]:
    """Labelled pairs with their readings, design weights and reference labels."""
    d = EVAL_DIR / "annotation" / batch
    key = json.loads((d / f"{batch}_key.json").read_text())
    ref = json.loads((d / "reference.json").read_text())["items"]
    datas: dict = {}
    rows = []
    for item in key["items"]:
        if item["item_id"] not in ref:
            continue  # fewer than three labels: left out, counted in the report
        k = (item["run"], item["topic"])
        if k not in datas:
            datas[k] = {r["pair_id"]: r for r in readings(load_topic(*k))}
        r = dict(datas[k][item["pair_id"]])
        assert r["stratum"] == item["stratum"], "the readings changed after the sample was drawn"
        r.update(item_id=item["item_id"], ref=ref[item["item_id"]], w=key["design"]["strata"][item["stratum"]]["weight"])
        r["w_c4"] = r["w"] if r["stratum"] in ("A", "B") else 1.0  # C4's estimator: every trigger pair, plus a sample of the rest unweighted
        r["opposed"] = r["ref"]["stance"] == "oppose"
        r["status"] = r["ref"]["status"]
        rows.append(r)
    return rows, key


def _prf(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> tuple[float | None, float | None, float | None]:
    tp, n_pred, n_pos = float((w * (y & pred)).sum()), float((w * pred).sum()), float((w * y).sum())
    return (tp / n_pred if n_pred else None, tp / n_pos if n_pos else None, 2 * tp / (n_pred + n_pos) if n_pred + n_pos else None)


def detector_metrics(rows: list[dict], name: str, boot: Boot) -> dict:
    y = np.array([r["opposed"] for r in rows])
    pred = np.array([r[name] for r in rows])
    w, wc = np.array([r["w"] for r in rows]), np.array([r["w_c4"] for r in rows])
    out: dict = {"flagged": int(pred.sum()), "true_positives": int((y & pred).sum()), "opposed_in_sample": int(y.sum())}
    for label, weights in (("design", w), ("c4", wc)):
        base = _prf(y, pred, weights)
        for k, key in enumerate(("precision", "recall", "f1")):
            out[f"{key}_{label}"] = {"value": base[k], "ci": boot.ci(lambda i, k=k, weights=weights: _prf(y[i], pred[i], weights[i])[k])}
    trig = np.array([r["current"] for r in rows])
    out["retention_within_trigger"] = {"value": _prf(y & trig, pred, w)[1], "ci": boot.ci(lambda i: _prf(y[i] & trig[i], pred[i], w[i])[1])}
    return out


def by_topic(rows: list[dict], name: str) -> dict:
    out = {}
    for t in sorted({r["topic"] for r in rows}):
        sub = [r for r in rows if r["topic"] == t]
        y, pred = np.array([r["opposed"] for r in sub]), np.array([r[name] for r in sub])
        p, _, _ = _prf(y, pred, np.array([r["w"] for r in sub]))
        _, rc, _ = _prf(y, pred, np.array([r["w_c4"] for r in sub]))
        out[t] = {"labelled": len(sub), "opposed": int(y.sum()), "flagged": int(pred.sum()), "true_positives": int((y & pred).sum()), "precision": p, "recall": rc}
    return out


def collapses(topic_stats: dict) -> list[str]:
    """Pre-registered: a topic with at least 5 flagged pairs and precision below 50%, or with at least 5 opposed
    labelled pairs and recall below 30%."""
    return [t for t, s in topic_stats.items()
            if (s["flagged"] >= 5 and (s["precision"] or 0) < 0.50) or (s["opposed"] >= 5 and (s["recall"] or 0) < 0.30)]


def gain_leave_one_topic_out(rows: list[dict], name: str) -> dict[str, float | None]:
    """Design-weighted precision gain over the current trigger with each topic removed in turn."""
    out = {}
    for t in sorted({r["topic"] for r in rows}):
        sub = [r for r in rows if r["topic"] != t]
        y, w = np.array([r["opposed"] for r in sub]), np.array([r["w"] for r in sub])
        a, b = _prf(y, np.array([r[name] for r in sub]), w)[0], _prf(y, np.array([r["current"] for r in sub]), w)[0]
        out[t] = (a - b) if (a is not None and b is not None) else None
    return out


def outcome(m: dict, topic_stats: dict, loto_gain: dict) -> tuple[str, list[str]]:
    """'success' | 'strong' | 'failure' | 'intermediate' for one E1 detector, by the pre-registered thresholds."""
    p, r = m["precision_design"]["value"], m["recall_c4"]["value"]
    notes = []
    collapsed = collapses(topic_stats)
    one_topic = any(v is not None and v <= 0 for v in loto_gain.values())
    if p is None or p < 0.65:
        notes.append("precision below 65%")
    if r is None or r < 0.50:
        notes.append("recall below 50%")
    if one_topic:
        notes.append("precision gain over the current trigger vanishes when one topic is removed")
    if notes:
        return "failure", notes
    if p >= 0.75 and r >= 0.60 and not collapsed:
        gain = m["precision_gain"]["value"]
        return ("strong" if r >= 0.70 and gain is not None and gain > 0 else "success"), notes
    if collapsed:
        notes.append(f"collapse on: {', '.join(collapsed)}")
    return "intermediate", notes


def verdict(results: dict) -> tuple[str, str]:
    """SUPPORTED | PROMISING BUT INCONCLUSIVE | FALSIFIED, by the pre-registered rule."""
    outs = {d: results["detectors"][d]["outcome"] for d in E1}
    if all(o == "failure" for o in outs.values()):
        return "FALSIFIED", "both E1 detectors meet a failure condition"
    for d in E1:
        m = results["detectors"][d]
        if outs[d] in ("success", "strong"):
            lo = m["precision_design"]["ci"][0]
            glo = m["precision_gain"]["ci"][0]
            if lo is not None and lo >= 0.65 and glo is not None and glo > 0:
                return "SUPPORTED", f"{DETECTORS[d]} meets the {outs[d]} criteria; its precision interval stays above 65% and its precision gain over the current trigger excludes zero"
    if any(o in ("success", "strong") for o in outs.values()):
        return "PROMISING BUT INCONCLUSIVE", "the point estimates meet the success criteria, but an interval reaches the failure zone or the gain over the current trigger does not exclude zero"
    return "PROMISING BUT INCONCLUSIVE", "no E1 detector meets the success criteria and at least one does not meet a failure condition"


def four_state(rows: list[dict], pred: list[str], boot: Boot) -> dict:
    gold = [r["status"] for r in rows]
    g, p, w = np.array(gold), np.array(pred), np.array([r["w"] for r in rows])
    res = FS.evaluate4(gold, pred)
    res["accuracy_ci"] = boot.ci(lambda i: float((g[i] == p[i]).mean()))
    res["macro_f1_ci"] = boot.ci(lambda i: FS.evaluate4(list(g[i]), list(p[i]))["macro_f1"])
    res["accuracy_weighted"] = float((w * (g == p)).sum() / w.sum())
    res["accuracy_weighted_ci"] = boot.ci(lambda i: float((w[i] * (g[i] == p[i])).sum() / w[i].sum()))
    return res


def cell(m: dict) -> str:
    return "–" if m["value"] is None else pct(m["value"], *m["ci"])


def run_validation(batch: str = "f1", out_dir: Path | None = None, *, n_boot: int = 2000, seed: int = 31) -> dict:
    rows, key = load(batch)
    boot = Boot([r["cluster"] for r in rows], n_boot, seed)
    y = np.array([r["opposed"] for r in rows])
    w = np.array([r["w"] for r in rows])
    res: dict = {"batch": batch, "labelled": len(rows), "sampled": len(key["items"]), "design": key["design"],
                 "opposed": int(y.sum()), "stance_majority_missing": sum(r["ref"]["stance"] is None for r in rows),
                 "status": dict(Counter(str(r["status"]) for r in rows)), "paper_pairs": len({r["cluster"] for r in rows}),
                 "by_stratum": {k: {"labelled": sum(r["stratum"] == k for r in rows), "opposed": sum(r["stratum"] == k and r["opposed"] for r in rows)} for k in STRATA},
                 "detectors": {}, "topics": {}}
    cur = np.array([r["current"] for r in rows])
    for name in DETECTORS:
        m = detector_metrics(rows, name, boot)
        pred = np.array([r[name] for r in rows])
        gain = (_prf(y, pred, w)[0] or 0) - (_prf(y, cur, w)[0] or 0) if pred.any() and cur.any() else None
        m["precision_gain"] = {"value": gain, "ci": boot.ci(lambda i, pred=pred: ((_prf(y[i], pred[i], w[i])[0] or 0) - (_prf(y[i], cur[i], w[i])[0] or 0)) if pred[i].any() and cur[i].any() else None)}
        m["recall_change"] = {"value": (_prf(y, pred, np.array([r["w_c4"] for r in rows]))[1] or 0) - (_prf(y, cur, np.array([r["w_c4"] for r in rows]))[1] or 0), "ci": (None, None)}
        res["topics"][name] = by_topic(rows, name)
        if name in E1:
            m["loto_gain"] = gain_leave_one_topic_out(rows, name)
            m["outcome"], m["outcome_notes"] = outcome(m, res["topics"][name], m["loto_gain"])
            m["collapsed_topics"] = collapses(res["topics"][name])
        res["detectors"][name] = m
    res["verdict"], res["verdict_reason"] = verdict(res)

    scored = [r for r in rows if r["status"]]
    sboot = Boot([r["cluster"] for r in scored], n_boot, seed + 1)
    res["end_to_end"] = {"n": len(scored)}
    systems = {"none": ("Constant NONE", lambda r: "NONE")}
    for name in ("current", "rel_ctx", "agree", "vote2"):
        systems[name] = (f"{DETECTORS[name]} → V1 gate", lambda r, name=name: FS.UP[r["v1"]] if r[name] else "NONE")
    for name, (_, fn) in systems.items():
        res["end_to_end"][name] = four_state(scored, [fn(r) for r in scored], sboot)

    # ------------------------------------------------------------------ report (sections A-D; E and F are added by hand after this file is frozen)
    d = key["design"]
    lines = [f"# Fresh-topic validation of the frozen E1 opposition detectors (batch {batch})", "",
             "Design, estimators, criteria and verdict rule were fixed in `eval/annotation/f1/PREREGISTRATION.md` before any pair was labelled. "
             "Reference: blinded model-assisted consensus labels under the C4 protocol, unadjudicated. Intervals: 95% bootstrap over paper pairs.", "",
             "## A. Dataset", "",
             f"{d['candidates']} candidate pairs from {len(d['pool_by_topic'])} fresh topics; {len(key['items'])} sampled, {len(rows)} with three labels "
             f"({len(rows) * 3} labels), from {res['paper_pairs']} paper pairs. Reference: {int(y.sum())} opposed at face value, {len(rows) - int(y.sum())} not. "
             "Status: " + ", ".join(f"{k} {v}" for k, v in sorted(res["status"].items())) + ".", "",
             md_table([["Stratum", "Definition", "Pool", "Labelled", "Weight", "Opposed by reference"],
                       *[[k, STRATA[k], str(d["strata"][k]["pool"]), str(res["by_stratum"][k]["labelled"]),
                          "–" if not d["strata"][k]["weight"] else f"{d['strata'][k]['weight']:.2f}", str(res["by_stratum"][k]["opposed"])] for k in STRATA]]), "",
             "Candidate pool by topic: " + "; ".join(f"{t}: {c}" for t, c in d["pool_by_topic"].items()) + ".", "",
             "## B. Frozen E1 validation", "",
             "Precision is design-weighted (an estimate for the whole candidate pool). Recall (C4-comparable) is taken as C4 took it: over the labelled "
             "sample, which holds the trigger-fired pairs and a sample of the others. Full-pool recall weights every stratum; C4 never measured it, and "
             "it is imprecise because stratum E is sampled thinly.", "",
             md_table([["Detector", "Flagged", "True positives", "Precision", "Recall (C4-comparable)", "F1 (C4-comparable)", "Recall, full pool", "Kept of the trigger's true positives", "Δ precision vs current"],
                       *[[DETECTORS[n], str(m["flagged"]), str(m["true_positives"]), cell(m["precision_design"]), cell(m["recall_c4"]), cell(m["f1_c4"]),
                          cell(m["recall_design"]), cell(m["retention_within_trigger"]),
                          "–" if m["precision_gain"]["value"] is None else f"{100 * m['precision_gain']['value']:+.1f}" + (f" [{100 * m['precision_gain']['ci'][0]:+.0f}, {100 * m['precision_gain']['ci'][1]:+.0f}]" if m["precision_gain"]["ci"][0] is not None else "")]
                         for n, m in res["detectors"].items()]]), "",
             "Pre-registered outcome per E1 detector: " + "; ".join(f"{DETECTORS[n]}: **{res['detectors'][n]['outcome']}**" + (f" ({'; '.join(res['detectors'][n]['outcome_notes'])})" if res['detectors'][n]['outcome_notes'] else "") for n in E1) + ".", "",
             "## C. Topic-level robustness", ""]
    table = [["Topic", "Labelled / opposed", *[f"{DETECTORS[n].split(':')[0].split(' (')[0]}: flagged / right, precision, recall" for n in ("current", "rel_ctx", "agree", "vote2")]]]
    for t in sorted({r["topic"] for r in rows}):
        s0 = res["topics"]["current"][t]
        table.append([t, f"{s0['labelled']} / {s0['opposed']}", *[f"{s['flagged']} / {s['true_positives']}, {pct(s['precision'])}, {pct(s['recall'])}" for s in (res["topics"][n][t] for n in ("current", "rel_ctx", "agree", "vote2"))]])
    lines += [md_table(table), ""]
    for n in E1:
        m = res["detectors"][n]
        lines.append(f"{DETECTORS[n]}: collapsed topics: {m['collapsed_topics'] or 'none'}; precision gain over the current trigger with one topic removed: "
                     + ", ".join(f"without {t} {'–' if v is None else f'{100 * v:+.1f}'}" for t, v in m["loto_gain"].items()) + ".")
    lines += ["", "## D. End to end", "",
              f"Four states on the {len(scored)} labelled pairs with a consensus status ({len(rows) - len(scored)} without one are left out).", "",
              md_table([["System", "Accuracy", "Accuracy, design-weighted", "Macro-F1", "GENUINE said / right", "CONTEXT_DEP. said / right", "INSUFFICIENT said / right", "NONE said / right"],
                        *[[label, pct(r["accuracy"], *r["accuracy_ci"]), pct(r["accuracy_weighted"], *r["accuracy_weighted_ci"]), pct(r["macro_f1"], *r["macro_f1_ci"]),
                           *[f"{r[s]['pred']} / {r[s]['tp']}" for s in ("GENUINE", "CONTEXT_DEPENDENT", "INSUFFICIENT_EVIDENCE", "NONE")]]
                          for label, r in ((systems[n][0], res["end_to_end"][n]) for n in systems)]]), "",
              md_table([["System", *[f"{FS.SHORT[s]}: precision / recall" for s in FS.STATES]],
                        *[[systems[n][0], *[f"{pct(res['end_to_end'][n][s]['precision'])} / {pct(res['end_to_end'][n][s]['recall'])}" for s in FS.STATES]] for n in systems]]), "",
              "## F. Verdict by the pre-registered rule", "", f"**{res['verdict']}** — {res['verdict_reason']}.", ""]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validation.md").write_text("\n".join(lines) + "\n")
        (out_dir / "validation.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    return res
