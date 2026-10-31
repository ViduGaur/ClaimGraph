"""Why apparent contradictions were not asserted.

Compares the contradiction detectors (NLI, direct LLM, ClaimGraph without its gate) with the full system:

1. what the full system did with each pair a detector calls a contradiction, and why;
2. for each face-value conflict, the gate's verdict and the context factors that decided it;
3. where reference labels exist, whether the verdicts are borne out by them;
4. representative examples with their explanations.

Parts 1, 2 and 4 describe system behaviour on all candidate pairs and need no labels.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evaluation.experiment import Systems, TopicData, predict, predict_full
from app.evaluation.metrics import ABSTAIN, RELATIONS
from app.evaluation.report import explanation_lines
from app.reasoning.relations import FACTORS

CATEGORY = {
    "dataset": "Dataset mismatch", "data scale": "Data-scale mismatch", "experimental conditions": "Experimental-condition mismatch",
    "training & design": "Training-setup mismatch", "task": "Task mismatch", "population": "Population mismatch",
    "evaluation metric": "Metric mismatch", "model / method": "Model / method mismatch", "baseline": "Baseline mismatch",
    "sample size": "Sample-size mismatch", "methodology": "Methodology mismatch", "time period": "Time-period mismatch",
}
SUBJECT = {"model / method", "baseline"}
DETECTORS = [("nli", "NLI only"), ("llm_direct_claims", "LLM direct (claims only)"), ("llm_direct_context", "LLM direct (+ passages)"),
             ("hybrid_no_gate", "ClaimGraph without context gate")]
OUTCOMES = ["asserted: genuine conflict", "gate: context-dependent (QUALIFIES)", "gate: insufficient evidence (not asserted)",
            "no face-value conflict: narrows / conditions (QUALIFIES)", "no face-value conflict: agreement (SUPPORTS / EXTENDS)",
            "no face-value conflict: different relationship (UNRELATED)"]


def full_outcome(label: str | None, verdict: str | None) -> str:
    if verdict == "genuine":
        return OUTCOMES[0]
    if verdict == "context_dependent":
        return OUTCOMES[1]
    if verdict == "insufficient":
        return OUTCOMES[2]
    if label == "QUALIFIES":
        return OUTCOMES[3]
    if label in ("SUPPORTS", "EXTENDS"):
        return OUTCOMES[4]
    return OUTCOMES[5]


def gate_reason(rel) -> tuple[str, list[str]]:
    """(mutually exclusive reason category, decisive factors) for one face-value conflict of the full system."""
    decisive = [f.factor for f in rel.factors if f.status == "different"]
    stated_both = [f.factor for f in rel.factors if f.status != "unknown"]
    return decisive, stated_both


def pct(k: int, n: int) -> str:
    return f"{k} ({100 * k / n:.0f}%)" if n else "0"


def md_table(rows: list[list[str]]) -> str:
    head, *body = rows
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|", *["| " + " | ".join(r) + " |" for r in body]])


def analyse(datas: dict[str, TopicData], out_dir: Path, gs=None) -> dict:
    gold_by_pair: dict[str, dict] = {}
    if gs is not None:
        gold_by_pair = {i["pair_id"]: gs.gold[i["item_id"]] for i in gs.key_items if i["item_id"] in gs.gold}
    detector_outcomes: dict[str, Counter] = {name: Counter() for name, _ in DETECTORS}
    conflicts: list[dict] = []
    n_pairs = 0
    for tid in sorted(datas):
        data = datas[tid]
        systems = Systems(data)
        for rec in data.records:
            n_pairs += 1
            label, verdict, rel = predict_full(systems, "hybrid_full", rec)
            outcome = full_outcome(label, verdict)
            for name, _ in DETECTORS:
                if predict(systems, name, rec)[0] == "CONTRADICTS":
                    detector_outcomes[name][outcome] += 1
            if verdict:
                decisive = [f.factor for f in rel.factors if f.status == "different"]
                stated_both = [f.factor for f in rel.factors if f.status != "unknown"]
                if verdict == "context_dependent":
                    category = CATEGORY[decisive[0]] if len(decisive) == 1 else "Multiple context mismatches"
                elif verdict == "insufficient":
                    category = ("Insufficient context: no factor stated by both papers" if not stated_both else
                                "Insufficient context: only the method / baseline is stated by both papers")
                else:
                    category = "Comparable context, no stated mismatch (contradiction asserted)"
                conflicts.append({"topic": tid, "pair_id": rec["pair_id"], "verdict": verdict, "label": label, "category": category,
                                  "decisive": decisive, "stated_both": stated_both, "rec": rec, "data": data, "rel": rel,
                                  "nli_contradiction": predict(systems, "nli", rec)[0] == "CONTRADICTS",
                                  "gold": gold_by_pair.get(rec["pair_id"])})
    n_conf = len(conflicts)
    by_verdict = Counter(c["verdict"] for c in conflicts)
    rejected = [c for c in conflicts if c["verdict"] != "genuine"]
    ctx = [c for c in conflicts if c["verdict"] == "context_dependent"]
    categories = Counter(c["category"] for c in rejected)
    factor_counts = Counter(f for c in ctx for f in c["decisive"])
    only_subject = sum(1 for c in ctx if set(c["decisive"]) <= SUBJECT)
    n_decisive = Counter(min(len(c["decisive"]), 4) for c in ctx)

    md = ["# Why apparent contradictions were not asserted", "",
          f"{n_pairs} candidate pairs from {len(datas)} topics. All counts in sections 1 and 2 describe system behaviour and "
          "do not depend on gold labels.", "",
          "## 1. What the full system does with pairs that a contradiction detector flags", "",
          "Each row: the pairs one detector calls CONTRADICTS, split by the full system's outcome for the same pairs.", ""]
    rows = [["Detector", "Flags", *OUTCOMES]]
    for name, title in DETECTORS:
        total = sum(detector_outcomes[name].values())
        rows.append([title, str(total), *[pct(detector_outcomes[name][o], total) for o in OUTCOMES]])
    md += [md_table(rows), "",
           "## 2. The full system's own face-value conflicts and the gate's verdict", "",
           f"The full system sees {n_conf} face-value conflicts (stance `oppose`, or `partial` with confident NLI contradiction).", "",
           md_table([["Gate verdict", "Pairs"],
                     ["context-dependent → relation QUALIFIES", pct(by_verdict["context_dependent"], n_conf)],
                     ["insufficient evidence → CONTRADICTS not asserted", pct(by_verdict["insufficient"], n_conf)],
                     ["genuine → relation CONTRADICTS", pct(by_verdict["genuine"], n_conf)]]), "",
           f"### Why the {len(rejected)} conflicts were not asserted (one reason per pair)", "",
           md_table([["Reason", "Pairs"], *[[k, pct(v, len(rejected))] for k, v in categories.most_common()]]), "",
           f"### Decisive context factors in the {len(ctx)} context-dependent verdicts (a pair can have several)", "",
           md_table([["Factor", "Pairs in which it is a decisive mismatch"],
                     *[[label, pct(factor_counts[label], len(ctx))] for label, _, _ in sorted(FACTORS, key=lambda f: -factor_counts[f[0]])
                       if factor_counts[label]]]), "",
           md_table([["Number of decisive mismatches", "Pairs"], *[[("4 or more" if k == 4 else str(k)), pct(v, len(ctx))] for k, v in sorted(n_decisive.items())]]), "",
           f"In {pct(only_subject, len(ctx))} of the context-dependent verdicts the only decisive mismatches are the method / "
           "model or the baseline, i.e. what the claims are about rather than the conditions they were tested under.", ""]

    summary: dict = {"pairs": n_pairs, "conflicts": n_conf, "verdicts": dict(by_verdict), "reasons": dict(categories),
                     "decisive_factors": dict(factor_counts), "n_decisive": dict(n_decisive), "only_subject_factors": only_subject,
                     "detector_outcomes": {k: dict(v) for k, v in detector_outcomes.items()}}

    if gs is not None:
        labelled = [c for c in conflicts if c["gold"]]
        md += ["## 3. Are the gate's verdicts borne out by the gold labels?", "",
               f"Gold: {gs.description}. {len(labelled)} of the {n_conf} face-value conflicts are in the gold set.", ""]
        table: dict[str, Counter] = defaultdict(Counter)
        for c in labelled:
            table[c["verdict"]][c["gold"]["label"]] += 1
        md += [md_table([["Gate verdict", "n", *RELATIONS],
                         *[[v, str(sum(table[v].values())), *[str(table[v][r]) for r in RELATIONS]] for v in ("context_dependent", "insufficient", "genuine")]]), ""]
        rej = [c for c in labelled if c["verdict"] != "genuine"]
        ok = sum(1 for c in rej if c["gold"]["label"] != "CONTRADICTS")
        cd = [c for c in labelled if c["verdict"] == "context_dependent"]
        md += [f"- Not asserted by the gate and not a contradiction in the gold labels: {pct(ok, len(rej))} of {len(rej)}.",
               f"- Context-dependent verdicts whose gold relation is QUALIFIES: {pct(sum(1 for c in cd if c['gold']['label'] == 'QUALIFIES'), len(cd))} of {len(cd)}; "
               f"UNRELATED: {pct(sum(1 for c in cd if c['gold']['label'] == 'UNRELATED'), len(cd))}; "
               f"CONTRADICTS (the gate rejected a gold contradiction): {pct(sum(1 for c in cd if c['gold']['label'] == 'CONTRADICTS'), len(cd))}.", ""]
        by_cat: dict[str, Counter] = defaultdict(Counter)
        for c in rej:
            by_cat[c["category"]][c["gold"]["label"]] += 1
        md += ["### Gold relation by rejection reason", "",
               md_table([["Reason", "n", *RELATIONS], *[[k, str(sum(v.values())), *[str(v[r]) for r in RELATIONS]]
                                                        for k, v in sorted(by_cat.items(), key=lambda kv: -sum(kv[1].values()))]]), ""]
        summary["gold_by_verdict"] = {v: dict(t) for v, t in table.items()}
        summary["gold_by_reason"] = {k: dict(v) for k, v in by_cat.items()}
        summary["rejected_not_gold_contradiction"] = {"k": ok, "n": len(rej)}

    md += ["## Representative examples (with the grounded explanation)", ""]
    shown = 0
    for category, _ in categories.most_common() + [("Comparable context, no stated mismatch (contradiction asserted)", 0)]:
        pool = [c for c in conflicts if c["category"] == category]
        # prefer pairs that NLI also calls a contradiction, that have an LLM explanation, and that the gold labels bear out
        def rank(c: dict) -> tuple:
            exp = c["rec"].get("explanation") or {}
            gold = (c["gold"] or {}).get("label")
            borne_out = gold == ("CONTRADICTS" if c["verdict"] == "genuine" else "QUALIFIES")
            return (-int(borne_out), -int(exp.get("explanation_source") == "llm"), -int(c["nli_contradiction"]), c["pair_id"])
        for c in sorted(pool, key=rank)[:2]:
            rec, data = c["rec"], c["data"]
            a, b = data.claims[rec["a"]], data.claims[rec["b"]]
            gold = f" · gold relation: {c['gold']['label']}" if c["gold"] else ""
            md += [f"### {category}", "",
                   f"`{rec['pair_id']}` · {c['topic']} · relation {'CONTRADICTS (not asserted)' if c['label'] == ABSTAIN else c['label']} · "
                   f"verdict {c['verdict']}{' · NLI also says contradiction' if c['nli_contradiction'] else ''}{gold}", "",
                   f"- **A** ({a.source_citation}): “{a.claim_text}”", f"- **B** ({b.source_citation}): “{b.claim_text}”", ""]
            if rec.get("explanation"):
                md += ["```", *explanation_lines(rec["explanation"], ""), "```", ""]
            shown += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rejection_analysis.md").write_text("\n".join(md))
    (out_dir / "rejection_analysis.json").write_text(json.dumps(summary, indent=1))
    return summary
