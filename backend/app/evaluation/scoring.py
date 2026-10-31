"""Score every system against one gold set and write tables, per-item predictions and plots.

All systems are scored on the same items and the same gold labels (asserted in :func:`build_examples`).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation import annotation as ann
from app.evaluation import metrics as M
from app.evaluation.experiment import SPEC_BY_NAME, SYSTEM_SPECS, Systems, TopicData, llm_view, predict_full, two_level
from app.reasoning.alignment import GroundingStats

MAIN_SYSTEMS = ["similarity", "nli", "llm_direct_claims", "llm_direct_context", "rules_full", "hybrid_no_gate", "hybrid_full"]
ABLATIONS = ["hybrid_no_gate", "hybrid_no_stance", "hybrid_no_nli", "hybrid_no_grounding", "rules_full"]
DEGENERATE = ["hybrid_always_qualify", "hybrid_always_abstain", "always_qualifies", "always_unrelated"]
MAIN_COLUMNS = [("false_con_per_100", "False con. /100 ↓"), ("con_precision", "Con-P"), ("con_recall", "Con-R"),
                ("con_f1", "Con-F1"), ("macro_f1", "Macro-F1"), ("accuracy", "Acc."), ("verdict_acc", "Verdict acc.")]
ABLATION_METRICS = ["false_con_per_100", "con_f1", "macro_f1", "accuracy", "verdict_acc"]
REFERENCE = "hybrid_full"


@dataclass
class GoldSet:
    gold: dict[str, dict]  # item_id -> {"label": relation, "verdict": conflict verdict or None, "source": ...}
    key_items: list[dict]
    kind: str  # 'model_consensus' | 'two_question'
    description: str
    agreement: dict = field(default_factory=dict)
    design: dict = field(default_factory=dict)  # stratum -> {"pool": n, "sampled": n}
    excluded: list[str] = field(default_factory=list)


def load_gold(labels_dir: Path, mode: str = "final") -> GoldSet:
    """Gold labels of an annotation batch.

    * a batch with ``gold.json`` holds model-assisted consensus labels (``mode='final'``: adjudicated, unanimous and
      provisional-majority labels; ``mode='unanimous'``: only pairs all annotator models agree on, plus adjudicated ones);
    * otherwise the batch holds two-question labels from the annotation tool.
    """
    key_items: list[dict] = []
    design: dict = {}
    for key_path in sorted(labels_dir.glob("*_key.json")):
        key = json.loads(key_path.read_text())
        key_items += key["items"]
        design.update(key.get("design", {}))
    gold_path = labels_dir / "gold.json"
    if gold_path.exists():
        from app.evaluation.model_annotation import agreement_report

        doc = json.loads(gold_path.read_text())
        gold, excluded = {}, []
        if mode.startswith("annotator:"):
            # robustness check: one annotator model's own labels as the reference
            who = mode.split(":", 1)[1]
            for iid, g in doc["items"].items():
                label = g["model_labels"].get(who)
                if label:
                    fvo = g["model_face_value_opposition"].get(who)
                    verdict = "genuine" if label == "CONTRADICTS" else ("context_dependent" if label == "QUALIFIES" and fvo else None)
                    gold[iid] = {"label": label, "verdict": verdict, "source": f"annotator {who}"}
            model = next(a["model"] for a in doc["annotators"] if a["id"] == who)
            return GoldSet(gold, key_items, "model_consensus", f"labels of one annotator model only ({model}) - robustness check, not the gold set",
                           agreement_report(doc), design, [])
        for iid, g in doc["items"].items():
            usable = g["final"] and (mode == "final" or g["final_source"] in ("unanimous", "adjudicated"))
            if usable:
                gold[iid] = {"label": g["final"], "verdict": g["conflict_verdict"], "source": g["final_source"]}
            else:
                excluded.append(iid)
        desc = doc["method"] + (" — only pairs with unanimous or adjudicated labels" if mode == "unanimous" else "")
        return GoldSet(gold, key_items, "model_consensus", desc, agreement_report(doc), design, excluded)
    labels = ann.load_labels(labels_dir)
    adjudicated = None
    for adj_path in (labels_dir / "adjudicated.json", labels_dir / "adjudicated.csv"):
        if adj_path.exists():
            adjudicated = ann.read_label_file(adj_path)
            break
    gold, unresolved = ann.gold_labels(labels, adjudicated)
    return GoldSet(gold, key_items, "two_question", "two-question labels from the annotation tool",
                   ann.agreement_report(labels), design, unresolved)


def build_examples(gs: GoldSet, datas: dict[str, TopicData], system_names: list[str]) -> dict[str, list[M.Example]]:
    """{system: examples} over the gold-labelled items, in one fixed item order, with design weights."""
    by_item = {i["item_id"]: i for i in gs.key_items}
    records = {tid: {r["pair_id"]: r for r in d.records} for tid, d in datas.items()}
    systems = {tid: Systems(d) for tid, d in datas.items()}
    scored = [iid for iid in sorted(gs.gold) if iid in by_item and by_item[iid]["topic"] in datas and gs.gold[iid].get("label")]
    per_stratum = Counter(by_item[iid]["stratum"] for iid in scored)
    out: dict[str, list[M.Example]] = {n: [] for n in system_names}
    for iid in scored:
        item, g = by_item[iid], gs.gold[iid]
        pool = (gs.design.get(item["stratum"]) or {}).get("pool")
        weight = pool / per_stratum[item["stratum"]] if pool else 1.0
        rec = records[item["topic"]][item["pair_id"]]
        for name in system_names:
            label, verdict, _ = predict_full(systems[item["topic"]], name, rec)
            out[name].append(M.Example(iid, item["topic"], g["label"], label, g.get("verdict"), verdict, weight, item["stratum"]))
    ids = [[e.item_id for e in ex] for ex in out.values()]
    golds = [[e.gold for e in ex] for ex in out.values()]
    assert all(x == ids[0] for x in ids) and all(x == golds[0] for x in golds), "systems must share items and gold labels"
    return out


def fmt(v: float | None, lo: float | None = None, hi: float | None = None) -> str:
    if v is None:
        return "–"
    s = f"{100 * v:.1f}"
    if lo is not None and hi is not None:
        s += f" [{100 * lo:.0f}, {100 * hi:.0f}]"
    return s


def title_of(name: str) -> str:
    return SPEC_BY_NAME[name].title


def main_table(results: dict[str, dict], names: list[str], *, ci: bool = True) -> list[list[str]]:
    rows = [["System", "Asserted con.", "Abstains", *[h for _, h in MAIN_COLUMNS]]]
    for name in names:
        r = results[name]
        row = [title_of(name), str(r["counts"]["n_pred_con"]), str(r["counts"]["n_abstain"])]
        for key, _ in MAIN_COLUMNS:
            m = r["metrics"][key]
            row.append(fmt(m["value"], m["lo"] if ci else None, m["hi"] if ci else None))
        rows.append(row)
    return rows


def per_class_table(results: dict[str, dict], names: list[str]) -> list[list[str]]:
    rows = [["System", *[f"{c[:5].title()} P/R/F1" for c in M.RELATIONS]]]
    for name in names:
        pc = results[name]["per_class"]
        rows.append([title_of(name).strip(), *[" / ".join(fmt(pc[c][k]) for k in ("precision", "recall", "f1")) for c in M.RELATIONS]])
    return rows


def to_markdown(rows: list[list[str]]) -> str:
    head, *body = rows
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def to_latex(rows: list[list[str]], caption: str, label: str) -> str:
    def esc(s: str) -> str:
        return s.replace("&", "\\&").replace("%", "\\%").replace("↓", "$\\downarrow$").replace("–", "--").replace("_", "\\_")

    head, *body = rows
    spec = "l" + "r" * (len(head) - 1)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(esc(h) for h in head) + " \\\\", "\\midrule"]
    lines += [" & ".join(esc(c) for c in r) + " \\\\" for r in body]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def paired_rows(examples: dict[str, list[M.Example]], names: list[str], n_boot: int) -> tuple[list[list[str]], dict]:
    """Paired differences (variant − full system) on the same items."""
    ref = examples[REFERENCE]
    rows = [["Variant", *[f"Δ {m}" for m in ABLATION_METRICS]]]
    raw: dict = {}
    for name in names:
        row, raw[name] = [title_of(name).strip()], {}
        for metric in ABLATION_METRICS:
            d = M.paired_bootstrap(examples[name], ref, M.METRIC_FNS[metric], n_boot=n_boot)
            raw[name][metric] = d
            if d is None:
                row.append("–")
            else:
                ci = f" [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]" if d["lo"] is not None else ""
                row.append(f"{100 * d['delta']:+.1f}{ci}{'*' if d['p'] is not None and d['p'] < 0.05 else ''}")
        rows.append(row)
    return rows, raw


def abstention_sensitivity(examples: dict[str, list[M.Example]], names: list[str]) -> list[list[str]]:
    rows = [["System", "Abstentions counted as", "Asserted con.", "False con. /100", "Con-P", "Con-R", "Con-F1", "Macro-F1", "Acc."]]
    for name in names:
        for reading, ex in (("abstention (primary)", examples[name]),
                            ("CONTRADICTS", M.remap_abstentions(examples[name], "CONTRADICTS")),
                            ("QUALIFIES", M.remap_abstentions(examples[name], "QUALIFIES"))):
            rows.append([title_of(name).strip(), reading, str(M.counts(ex)["n_pred_con"]),
                         *[fmt(M.METRIC_FNS[k](ex)) for k in ("false_con_per_100", "con_precision", "con_recall", "con_f1", "macro_f1", "accuracy")]])
    return rows


def gate_discrimination(gs: GoldSet, datas: dict[str, TopicData]) -> dict:
    """On the sampled face-value conflicts of the full system: the gate's verdict against the gold relation."""
    by_item = {i["item_id"]: i for i in gs.key_items}
    records = {tid: {r["pair_id"]: r for r in d.records} for tid, d in datas.items()}
    systems = {tid: Systems(d) for tid, d in datas.items()}
    table: dict[str, Counter] = defaultdict(Counter)
    for iid, g in gs.gold.items():
        item = by_item.get(iid)
        if item is None or item["topic"] not in datas:
            continue
        _, verdict, _ = predict_full(systems[item["topic"]], REFERENCE, records[item["topic"]][item["pair_id"]])
        if verdict:
            table[verdict][g["label"]] += 1
    out: dict = {v: {r: table[v][r] for r in M.RELATIONS} | {"n": sum(table[v].values())} for v in M.VERDICTS}
    # how often is a verdict borne out by the gold relation?
    out["rejected_is_not_contradiction"] = _share(table, ("context_dependent", "insufficient"), lambda r: r != "CONTRADICTS")
    out["context_dependent_is_qualifies"] = _share(table, ("context_dependent",), lambda r: r == "QUALIFIES")
    out["genuine_is_contradiction"] = _share(table, ("genuine",), lambda r: r == "CONTRADICTS")
    # does the verdict carry information about the gold relation? share of gold contradictions per verdict, and
    # Fisher's exact test of context-dependent verdicts against the rest
    con = {v: table[v]["CONTRADICTS"] for v in M.VERDICTS}
    tot = {v: sum(table[v].values()) for v in M.VERDICTS}
    out["gold_contradiction_share"] = {v: (con[v] / tot[v] if tot[v] else None) for v in M.VERDICTS}
    rest_con, rest_tot = con["genuine"] + con["insufficient"], tot["genuine"] + tot["insufficient"]
    out["fisher_context_dependent_vs_rest"] = fisher_exact_two_sided(con["context_dependent"], tot["context_dependent"] - con["context_dependent"],
                                                                     rest_con, rest_tot - rest_con)
    out["fisher_context_dependent_vs_insufficient"] = fisher_exact_two_sided(
        con["context_dependent"], tot["context_dependent"] - con["context_dependent"], con["insufficient"], tot["insufficient"] - con["insufficient"])
    return out


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]]."""
    from math import comb

    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0
    prob = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)  # noqa: E731
    observed = prob(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def _share(table: dict[str, Counter], verdicts: tuple[str, ...], ok) -> dict:
    n = sum(sum(table[v].values()) for v in verdicts)
    k = sum(c for v in verdicts for r, c in table[v].items() if ok(r))
    return {"k": k, "n": n, "share": k / n if n else None}


def nli_sweep(examples: list[M.Example], gs: GoldSet, datas: dict[str, TopicData],
              thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)) -> list[list[str]]:
    """The NLI-only baseline at several decision thresholds, so its comparison does not hinge on one cut-off."""
    by_item = {i["item_id"]: i for i in gs.key_items}
    records = {tid: {r["pair_id"]: r for r in d.records} for tid, d in datas.items()}
    systems = {tid: Systems(d) for tid, d in datas.items()}
    rows = [["NLI threshold", "Asserted con.", "False con. /100", "Con-P", "Con-R", "Con-F1", "Macro-F1"]]
    for t in thresholds:
        ex = []
        for e in examples:
            item = by_item[e.item_id]
            label, _ = systems[item["topic"]].nli_only(records[item["topic"]][item["pair_id"]], t)
            ex.append(M.Example(e.item_id, e.topic, e.gold, label, e.gold_verdict, "genuine" if label == "CONTRADICTS" else None))
        rows.append([f"{t:.1f}", str(M.counts(ex)["n_pred_con"]),
                     *[fmt(M.METRIC_FNS[k](ex)) for k in ("false_con_per_100", "con_precision", "con_recall", "con_f1", "macro_f1")]])
    return rows


def grounding_summary(datas: dict[str, TopicData], pair_ids: set[str] | None = None) -> dict:
    stats = GroundingStats()
    n = 0
    for d in datas.values():
        for rec in d.records:
            if pair_ids is not None and rec["pair_id"] not in pair_ids:
                continue
            if rec.get("llm_pair"):
                n += 1
                llm_view(d, rec, stats=stats)
    out = stats.as_dict()
    out["pairs"] = n
    out["dropped_rate"] = round(stats.dropped / stats.values, 4) if stats.values else None
    return out


def score(gs: GoldSet, datas: dict[str, TopicData], out_dir: Path, *, n_boot: int = 2000, title: str = "") -> dict:
    names = [s.name for s in SYSTEM_SPECS]
    weighted = build_examples(gs, datas, names)
    examples = {n: M.unweighted(ex) for n, ex in weighted.items()}
    some = examples[names[0]]
    n_items = len(some)
    results = {name: M.evaluate(ex, n_boot=n_boot) for name, ex in examples.items()}
    has_design = bool(gs.design)
    pool = {name: {k: M.METRIC_FNS[k](ex) for k in ("false_con_per_100", "accuracy", "macro_f1", "con_recall", "abstention")}
            for name, ex in weighted.items()} if has_design else {}
    pool_ci = {name: {k: M.bootstrap(weighted[name], M.METRIC_FNS[k], n_boot=n_boot) for k in ("false_con_per_100", "accuracy")}
               for name in MAIN_SYSTEMS} if has_design else {}
    abl_rows, abl_raw = paired_rows(examples, ABLATIONS, n_boot)
    deg_rows, deg_raw = paired_rows(examples, DEGENERATE, n_boot)
    base_rows, base_raw = paired_rows(examples, ["nli", "llm_direct_claims", "llm_direct_context"], n_boot)
    per_topic = {}
    for topic in sorted({e.topic for e in some}):
        per_topic[topic] = {name: {k: M.METRIC_FNS[k]([e for e in ex if e.topic == topic]) for k in ("false_con_per_100", "macro_f1", "accuracy")}
                            | M.counts([e for e in ex if e.topic == topic]) for name, ex in examples.items()}
    pair_ids = {i["pair_id"] for i in gs.key_items if i["item_id"] in gs.gold}
    summary = {
        "title": title, "gold": gs.description, "gold_kind": gs.kind, "n_items": n_items, "excluded_items": len(gs.excluded),
        "gold_distribution": M.label_distribution([e.gold for e in some]),
        "gold_verdicts": dict(Counter(e.gold_verdict for e in some if e.gold_verdict)),
        "gold_sources": dict(Counter(gs.gold[e.item_id].get("source", "?") for e in some)),
        "strata": {s: {"scored": sum(1 for e in some if e.stratum == s), **gs.design.get(s, {})} for s in sorted({e.stratum for e in some})},
        "topics": sorted({e.topic for e in some}), "agreement": gs.agreement,
        "results": results, "pool_estimates": pool, "paired_vs_full": {**abl_raw, **deg_raw, **base_raw}, "per_topic": per_topic,
        "gate_discrimination": gate_discrimination(gs, datas), "grounding": grounding_summary(datas, pair_ids),
        "confusion": {n: M.confusion(examples[n]) for n in names},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1))

    md = [f"# {title or 'Results'}", "",
          f"Gold labels: {gs.description}.",
          f"{n_items} pairs from {len(summary['topics'])} topics" + (f" ({len(gs.excluded)} sampled pairs without a usable gold label are excluded)" if gs.excluded else "") + ".",
          f"Gold relations: {summary['gold_distribution']}. Gold conflict verdicts: {summary['gold_verdicts'] or 'n/a'}. "
          f"Label sources: {summary['gold_sources']}.", "",
          f"Brackets are 95% bootstrap intervals ({n_boot} draws, items resampled within sampling strata). ‘–’ = undefined "
          "(for example a precision when nothing is asserted). An *abstention* is a face-value conflict the system declines "
          "to assert (relation CONTRADICTS flagged uncertain, verdict insufficient); it counts as wrong for accuracy and "
          "F1 and is never an asserted contradiction.", "",
          "## Main comparison (evaluation sample)", "", to_markdown(main_table(results, MAIN_SYSTEMS)), "",
          "## Per-class precision / recall / F1", "", to_markdown(per_class_table(results, MAIN_SYSTEMS)), "",
          "## Ablations (absolute scores)", "", to_markdown(main_table(results, [REFERENCE, *ABLATIONS, "rules_no_gate", "rules_no_stance"])), "",
          "## Ablations: paired difference to the full system (percentage points; * = bootstrap p < 0.05)", "", to_markdown(abl_rows), "",
          "## Baselines: paired difference to the full system", "", to_markdown(base_rows), "",
          "## Is the gate just being conservative? Degenerate alternatives", "",
          to_markdown(main_table(results, [REFERENCE, *DEGENERATE])), "", to_markdown(deg_rows), ""]
    gd = summary["gate_discrimination"]
    md += ["### Gate verdict against gold relation (sampled face-value conflicts of the full system)", "",
           to_markdown([["Gate verdict", "n", *M.RELATIONS], *[[v, str(gd[v]["n"]), *[str(gd[v][r]) for r in M.RELATIONS]] for v in M.VERDICTS]]), "",
           f"- conflicts the gate did not assert (context-dependent or insufficient) whose gold relation is indeed not CONTRADICTS: "
           f"{gd['rejected_is_not_contradiction']['k']}/{gd['rejected_is_not_contradiction']['n']}",
           f"- 'context-dependent' verdicts whose gold relation is QUALIFIES: {gd['context_dependent_is_qualifies']['k']}/{gd['context_dependent_is_qualifies']['n']}",
           f"- 'genuine' verdicts whose gold relation is CONTRADICTS: {gd['genuine_is_contradiction']['k']}/{gd['genuine_is_contradiction']['n']}", ""]
    share = gd["gold_contradiction_share"]
    md += ["Share of gold contradictions by gate verdict: " + ", ".join(f"{v} {fmt(share[v])}%" for v in M.VERDICTS)
           + f". Fisher's exact test, context-dependent versus insufficient: p = {gd['fisher_context_dependent_vs_insufficient']:.4f}; "
           f"context-dependent versus the other two verdicts: p = {gd['fisher_context_dependent_vs_rest']:.4f}.", ""]
    from app.evaluation.experiment import V2_SYSTEMS

    if V2_SYSTEMS:
        v2_rows, v2_raw = paired_rows(examples, V2_SYSTEMS, n_boot)
        summary["paired_vs_full"].update(v2_raw)
        md += ["## V2 comparability model against the V1 gate", "",
               "Same stance and factor tables; only the decision on a face-value conflict differs. Parameters are frozen in "
               "`eval/v2_params.json` (see its provenance for what they were calibrated on). **If they were calibrated on this batch, "
               "these rows are in-sample and are not a result.**", "",
               to_markdown(main_table(results, ["hybrid_no_gate", REFERENCE, *V2_SYSTEMS])), "", to_markdown(v2_rows), "",
               to_markdown(abstention_sensitivity(examples, [REFERENCE, *V2_SYSTEMS])), ""]
    md += ["## How abstentions are scored: sensitivity", "", to_markdown(abstention_sensitivity(examples, [REFERENCE, "rules_full"])), ""]
    # the two-level reading: an abstention is still a CONTRADICTS relation, only flagged uncertain
    alt = {n: M.remap_abstentions(ex, "CONTRADICTS") for n, ex in examples.items()}
    alt_results = {n: M.evaluate(alt[n], n_boot=n_boot) for n in [*MAIN_SYSTEMS, "hybrid_always_qualify", "hybrid_always_abstain"]}
    alt_rows, alt_raw = paired_rows(alt, ["hybrid_no_gate", "hybrid_always_qualify", "hybrid_always_abstain", "nli",
                                           "llm_direct_claims", "llm_direct_context"], n_boot)
    summary["relation_level_reading"] = {"results": alt_results, "paired_vs_full": alt_raw}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
    md += ["### Relation-level reading: an abstention counted as the CONTRADICTS relation it is filed under", "",
           "In the two-level representation an unasserted conflict has relation CONTRADICTS (flagged uncertain) and verdict "
           "insufficient. The tables above score it as an abstention, which was fixed as the primary reading before the gold "
           "labels existed. This table scores the relation label instead. The gold labels have no 'cannot tell' category, so "
           "the two readings differ a great deal; both are reported.", "",
           to_markdown(main_table(alt_results, [*MAIN_SYSTEMS, "hybrid_always_qualify", "hybrid_always_abstain"])), "",
           to_markdown(alt_rows), ""]
    if has_design:
        rows = [["System", "False con. /100 pairs (pool)", "Accuracy (pool)", "Macro-F1 (pool)", "Con-R (pool)"]]
        for name in MAIN_SYSTEMS:
            p = pool[name]
            rows.append([title_of(name), fmt(p["false_con_per_100"], *pool_ci[name]["false_con_per_100"]),
                         fmt(p["accuracy"], *pool_ci[name]["accuracy"]), fmt(p["macro_f1"]), fmt(p["con_recall"])])
        md += ["## Estimates for the whole candidate pool (design-weighted)", "",
               "The sample over-represents conflicts. Weighting each item by stratum size / items scored estimates the same "
               "quantities over all candidate pairs the sample was drawn from. Strata with few sampled items make these noisy.", "",
               to_markdown(rows), "",
               to_markdown([["Stratum", "Pool", "Scored"], *[[s, str(v.get("pool", "")), str(v["scored"])] for s, v in summary["strata"].items()]]), ""]
    md += ["## Per topic (false con. /100 · macro-F1 · accuracy)", ""]
    rows = [["Topic", "N", "gold con.", *[title_of(n).strip() for n in ("nli", "llm_direct_context", "hybrid_no_gate", "hybrid_full")]]]
    for topic, by in per_topic.items():
        rows.append([topic, str(by["nli"]["n"]), str(by["nli"]["n_gold_con"]),
                     *[" · ".join(fmt(by[n][k]) for k in ("false_con_per_100", "macro_f1", "accuracy")) for n in ("nli", "llm_direct_context", "hybrid_no_gate", "hybrid_full")]])
    md += [to_markdown(rows), "", "## NLI-only baseline across thresholds", "", to_markdown(nli_sweep(some, gs, datas)), "",
           "## Grounding of the LLM context table (scored pairs)", "", "```", json.dumps(summary["grounding"], indent=1), "```", ""]
    for name in MAIN_SYSTEMS[1:] + ["hybrid_always_qualify"]:
        table = summary["confusion"][name]
        md += [f"## Confusion — {title_of(name).strip()} (rows = gold, columns = predicted)", "",
               to_markdown([["gold \\ pred", *M.PRED_LABELS], *[[g, *[str(table[g][p]) for p in M.PRED_LABELS]] for g in M.RELATIONS]]), ""]
    if gs.agreement:
        md += ["## Annotation agreement", "", "```", json.dumps(gs.agreement, indent=1), "```", ""]
    (out_dir / "results.md").write_text("\n".join(md))
    (out_dir / "tables.tex").write_text(
        to_latex(main_table(results, MAIN_SYSTEMS, ci=False), "Main comparison on held-out topics.", "tab:main") + "\n\n"
        + to_latex(abl_rows, "Ablations: paired difference to the full system (percentage points).", "tab:ablation") + "\n\n"
        + to_latex(per_class_table(results, MAIN_SYSTEMS), "Per-class precision / recall / F1.", "tab:perclass") + "\n")
    with (out_dir / "per_item.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "topic", "stratum", "gold_relation", "gold_conflict_verdict", "gold_source",
                    *[c for n in names for c in (f"{n}:relation", f"{n}:conflict_verdict")]])
        for k, e in enumerate(some):
            cells = []
            for n in names:
                p = examples[n][k]
                relation, verdict = two_level(p.pred, p.pred_verdict)
                cells += [relation or "", verdict or ""]
            w.writerow([e.item_id, e.topic, e.stratum, e.gold, e.gold_verdict or "", gs.gold[e.item_id].get("source", ""), *cells])
    with (out_dir / "confusion_matrices.json").open("w") as fh:
        json.dump(summary["confusion"], fh, indent=1)
    try:
        from app.evaluation.plots import make_plots

        make_plots(summary, out_dir)
    except Exception as exc:  # noqa: BLE001 — plots are a convenience; never fail scoring over them
        (out_dir / "plots_skipped.txt").write_text(f"plots not generated: {exc}\n")
    return summary
