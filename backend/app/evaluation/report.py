"""Plain-text report of one prepared topic: claims, contexts, candidate pairs, every signal,
the decision and its evidence. This is the terminal view of what the claim graph is built from."""
from __future__ import annotations

import textwrap
from collections import Counter

from app.evaluation.experiment import SPEC_BY_NAME, Systems, TopicData, predict
from app.evaluation.metrics import ABSTAIN
from app.reasoning.conflicts import analyze_conflicts
from app.reasoning.explanations import explanation_artifact
from app.schemas.domain import Claim, ClaimContext, FactorComparison, Relationship

LABEL_ORDER = ["CONTRADICTS", "QUALIFIES", ABSTAIN, "EXTENDS", "SUPPORTS", "UNRELATED", None]
VERDICT_TEXT = {"CONTRADICTS": "genuine", "QUALIFIES": "context-dependent", ABSTAIN: "insufficient evidence"}


def headline(label: str | None, rel: Relationship | None) -> str:
    """'relation=… conflict_verdict=…': the two levels, with an abstention shown as an uncertain CONTRADICTS."""
    if label is None:
        return "NO PREDICTION"
    relation = "CONTRADICTS (uncertain, not asserted)" if label == ABSTAIN else label
    conflict = rel is not None and rel.signals.conflict_signal
    return f"relation={relation}" + (f"   conflict_verdict={VERDICT_TEXT[label]}" if conflict and label in VERDICT_TEXT else "")


def explanation_lines(record: dict, indent: str = "    ") -> list[str]:
    """The grounded explanation record of a face-value conflict, as text."""
    out = [_wrap("apparent conflict: " + record["apparent_conflict"], indent)]
    for row in record["decisive_mismatches"]:
        out.append(f"{indent}decisive mismatch — {row['factor']}: A = {row['a_value']}  |  B = {row['b_value']}")
        out.append(_wrap(f"A quote: “{row['a_quote'] or '-'}”   B quote: “{row['b_quote'] or '-'}”", indent + "    "))
    if not record["decisive_mismatches"] and record["comparable_factors"]:
        out.append(f"{indent}comparable factors: " + "; ".join(f"{r['factor']} (A = {r['a_value']} | B = {r['b_value']})" for r in record["comparable_factors"]))
    if record["one_sided_factors"]:
        out.append(f"{indent}stated by one paper only: " + "; ".join(f"{r['factor']} ({r['stated_by']}: {r['value']})" for r in record["one_sided_factors"]))
    out.append(_wrap(f"explanation [{record['explanation_source']}]: {record['explanation']}", indent))
    if record.get("explanation_note"):
        out.append(_wrap("note: " + record["explanation_note"], indent))
    return out


def _wrap(text: str, indent: str, width: int = 118) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def context_line(ctx: ClaimContext, inherited: list[str]) -> str:
    parts = []
    for key, value in ctx.model_dump().items():
        if not value:
            continue
        shown = ", ".join(value) if isinstance(value, list) else str(value)
        parts.append(f"{key}{'*' if key in inherited else ''}={shown}")
    return "; ".join(parts) or "not stated"


def format_claims(data: TopicData) -> str:
    out = [f"CLAIMS  ({len(data.claims)} from {len(data.papers)} papers; * = context inherited from paper-level text)"]
    by_paper: dict[str, list[Claim]] = {}
    for c in data.claims.values():
        by_paper.setdefault(c.paper_id, []).append(c)
    for pid, claims in by_paper.items():
        p = data.papers[pid]
        out.append(f"\n  {p.citation} — {p.title}  [{p.text_source}]")
        for c in claims:
            out.append(f"    {c.claim_id}  {c.claim_type.value}/{c.polarity.value}  section={c.section}  "
                       f"subject={c.subject or '-'}  conf={c.confidence}")
            out.append(_wrap(f"“{c.claim_text}”", "        "))
            out.append(_wrap("context: " + context_line(c.context, c.inherited_context), "        "))
    return "\n".join(out)


def factor_lines(factors: list[FactorComparison], indent: str) -> list[str]:
    lines = []
    for f in factors:
        if f.status == "unknown" and not (f.a or f.b):
            continue
        a, b = ", ".join(f.a) or "not stated", ", ".join(f.b) or "not stated"
        flags = (" (paper-level)" if f.inherited else "") + (f" [{f.note}]" if f.note else "")
        lines.append(f"{indent}{f.factor:24} {f.status.upper():9} A: {a}  |  B: {b}{flags}")
        if f.a_quote or f.b_quote:
            lines.append(_wrap(f"quotes — A: “{f.a_quote or '-'}”  B: “{f.b_quote or '-'}”", indent + " " * 4))
    return lines


def format_pair(data: TopicData, rec: dict, label: str | None, rel: Relationship | None, *, verbose: bool = True) -> str:
    a, b = data.claims[rec["a"]], data.claims[rec["b"]]
    con = (rec["nli_ab"]["contradiction"] + rec["nli_ba"]["contradiction"]) / 2
    ent = max(rec["nli_ab"]["entailment"], rec["nli_ba"]["entailment"])
    head = f"{rec['pair_id']}  {headline(label, rel)}   sim={rec['similarity']:.2f}  NLI con={con:.2f} ent={ent:.2f}"
    if rel is not None:
        head += f"  conf={rel.confidence}{' uncertain' if rel.uncertain else ''}  decided_by={rel.decided_by}"
    out = [head, _wrap(f"A {a.source_citation}: “{a.claim_text}”", "    "), _wrap(f"B {b.source_citation}: “{b.claim_text}”", "    ")]
    if rel is not None and verbose:
        out.extend(factor_lines(rel.factors, "      "))
        out.append(_wrap("why: " + rel.reasoning, "    "))
        if rec.get("explanation"):
            out.extend(explanation_lines(rec["explanation"]))
        else:
            conflict = analyze_conflicts([rel], data.claims)
            if conflict:
                out.append(_wrap("verdict: " + conflict[0].explanation, "    "))
            if rel.llm_note:
                out.append(_wrap("LLM stance note: " + rel.llm_note, "    "))
    return "\n".join(out)


def format_topic(data: TopicData, system: str = "hybrid_full", *, show_unrelated: bool = False, limit: int | None = None,
                 claims: bool = True) -> str:
    systems = Systems(data)
    rows = [(rec, *predict(systems, system, rec)) for rec in data.records]
    if all(label is None for _, label, _ in rows) and system.startswith("hybrid"):
        system = "rules_full"  # no LLM signals in this run yet
        rows = [(rec, *predict(systems, system, rec)) for rec in data.records]
    counts = Counter(label for _, label, _ in rows)
    out = [f"TOPIC {data.topic.id} [{data.topic.split}]: {data.topic.question}",
           f"system: {SPEC_BY_NAME[system].title.strip()} ({system})   pairs: {len(rows)}   "
           + "  ".join(f"{k}={counts[k]}" for k in LABEL_ORDER if counts.get(k)), ""]
    if claims:
        out += [format_claims(data), ""]
    out.append("CANDIDATE PAIRS")
    rows.sort(key=lambda r: (LABEL_ORDER.index(r[1]), -r[0]["similarity"]))
    shown = 0
    for rec, label, rel in rows:
        if label == "UNRELATED" and not show_unrelated:
            continue
        if limit is not None and shown >= limit:
            break
        out += ["", format_pair(data, rec, label, rel)]
        shown += 1
    if not show_unrelated and counts.get("UNRELATED"):
        out += ["", f"({counts['UNRELATED']} candidate pairs judged UNRELATED are hidden; pass --all to list them)"]
    return "\n".join(out)


def format_analysis(result) -> str:
    """The same terminal view for a live pipeline run (an ``AnalysisResult``)."""
    claims = {c.claim_id: c for c in result.claims}
    papers = {p.paper_id: p for p in result.papers}
    conflicts = {c.relation_id: c for c in result.conflicts}
    run = result.run
    out = [f"QUESTION: {result.query_plan.question}",
           f"{len(papers)} papers · {len(claims)} claims · {len(result.relationships)} relationships · "
           f"{len(conflicts)} apparent conflicts   [LLM: {run.llm_provider}{'/' + run.llm_model if run.llm_model else ''}, "
           f"{run.llm_calls} calls · NLI: {run.nli_backend} · {run.duration_s}s]", ""]
    out.append(f"CLAIMS  (* = context inherited from paper-level text)")
    for pid, p in papers.items():
        own = [c for c in claims.values() if c.paper_id == pid]
        out.append(f"\n  {p.citation} — {p.title}  [{p.text_source}]")
        for c in own:
            out.append(f"    {c.claim_id}  {c.claim_type.value}/{c.polarity.value}  section={c.section}  via={c.extraction_method}  conf={c.confidence}")
            out.append(_wrap(f"“{c.claim_text}”", "        "))
            out.append(_wrap("context: " + context_line(c.context, c.inherited_context), "        "))
    out += ["", "RELATIONSHIPS"]
    verdict_text = {"genuine_disagreement": "genuine", "context_dependent": "context-dependent", "insufficient_evidence": "insufficient evidence"}
    order = {"CONTRADICTS": 0, "QUALIFIES": 1, "EXTENDS": 2, "SUPPORTS": 3}
    for rel in sorted(result.relationships, key=lambda r: (order[r.relation.value], -r.confidence)):
        a, b = claims[rel.target_claim_id], claims[rel.source_claim_id]
        conflict = conflicts.get(rel.relation_id)
        verdict = f"   conflict_verdict={verdict_text[conflict.verdict.value]}" if conflict else ""
        relation = rel.relation.value + (" (uncertain, not asserted)" if rel.relation.value == "CONTRADICTS" and rel.uncertain else "")
        sig = rel.signals
        out += ["", f"{rel.relation_id}  relation={relation}{verdict}   sim={sig.similarity:.2f}  NLI con={sig.nli_contradiction:.2f} "
                    f"ent={sig.nli_entailment:.2f}  conf={rel.confidence}{' uncertain' if rel.uncertain else ''}  decided_by={rel.decided_by}",
                _wrap(f"A {a.source_citation}: “{a.claim_text}”", "    "), _wrap(f"B {b.source_citation}: “{b.claim_text}”", "    ")]
        out.extend(factor_lines(rel.factors, "      "))
        out.append(_wrap("why: " + rel.reasoning, "    "))
        if conflict:
            out.extend(explanation_lines(explanation_artifact(conflict, rel, a, b)))
        elif rel.llm_note:
            out.append(_wrap("LLM stance note: " + rel.llm_note, "    "))
    for w in getattr(result, "warnings", []) or []:
        out.append(f"warning: {w}")
    return "\n".join(out)
