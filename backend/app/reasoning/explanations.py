"""Grounded explanations of conflict verdicts.

The verdict on a face-value conflict (genuine / context-dependent / insufficient evidence) is made
by the rules in :mod:`app.reasoning.relations` and :mod:`app.reasoning.conflicts`. This module turns
that decision into something a reader can check:

* a **structured record**, built by code from the factor table: the apparent conflict, the relation
  and verdict, and for every decisive mismatch the factor, each paper's value and each paper's
  verbatim quote. Nothing in it comes from the explanation model;
* a short **prose explanation** written by the LLM from that record alone.

The LLM does not decide anything here and is not allowed to supply reasons. Its text is rejected, and
the deterministic template explanation kept, when it

* cites a factor that is not in the record, or (for a context-dependent verdict) cites none of the
  decisive ones;
* names one of the twelve context factors that the record does not contain;
* contains a number that appears nowhere in the claims, values or quotes;
* refers to "the table" instead of the papers, or (context-dependent verdicts) states no decisive mismatch with
  both papers' values as they stand in the record;
* is empty or longer than :data:`MAX_WORDS`.

A rejected text gets one retry in which the model is told what was wrong.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from app.providers.llm import BudgetedLLM
from app.reasoning.relations import FACTORS
from app.schemas.domain import Claim, ConflictAnalysis, ConflictVerdict, FactorComparison, Relationship

log = logging.getLogger(__name__)

MAX_WORDS = 130
VERDICT_NAME = {ConflictVerdict.GENUINE: "genuine", ConflictVerdict.CONTEXT_DEPENDENT: "context_dependent",
                ConflictVerdict.INSUFFICIENT_EVIDENCE: "insufficient"}

# distinctive words by which a factor would be named in prose
FACTOR_WORDS: dict[str, tuple[str, ...]] = {
    "data scale": ("data scale", "dataset size", "amount of data"),
    "dataset": ("dataset", "benchmark", "corpus"),
    "experimental conditions": ("experimental condition",),
    "training & design": ("training setup", "training design", "training & design"),
    "task": ("different task", "task differ"),
    "population": ("population",),
    "evaluation metric": ("metric",),
    "model / method": ("model / method", "architecture"),
    "baseline": ("baseline",),
    "sample size": ("sample size",),
    "methodology": ("methodology", "study type"),
    "time period": ("time period",),
}
assert set(FACTOR_WORDS) == {label for label, _, _ in FACTORS}

SYSTEM = (
    "You explain a decision that a rule-based system has ALREADY made about two claims from different scientific "
    "papers. You do not make, question or change the decision.\n"
    "Your reader sees two claims that look contradictory and wants to know how the pair was classified and on what "
    "evidence. Use ONLY what is in the user message: the two claims, the decision, and the context table (factor, "
    "value for A, value for B, quotes). Rules:\n"
    "- Name a context factor only if it is in the table, and use the values given there.\n"
    "- Do not add any cause, mechanism or scientific reason for why the results differ. You may state that the stated "
    "conditions differ; you may not say why that would change the outcome.\n"
    "- No outside knowledge about the papers, methods or datasets. No numbers that are not in the user message.\n"
    "- Do not say that either paper is right or wrong.\n"
    "- Write for a reader who sees only your text: refer to 'Paper A' and 'Paper B' and to the factors by name, never "
    "to 'the table' or 'the user message'.\n"
    "- When you mention a factor that differs, give Paper A's value and Paper B's value exactly as written in the "
    "table: copy the values, do not paraphrase or merge them. If more than three factors differ, give the three most "
    "relevant ones with their values and say how many others differ.\n"
    "What to write, by decision:\n"
    '- "context_dependent": the claims oppose at face value, but the table lists stated conditions that differ. Say '
    "which factors differ (A's value versus B's value) and conclude that the findings were not obtained under "
    "equivalent conditions, so they do not establish a direct contradiction.\n"
    '- "genuine": the claims oppose at face value and no stated condition differs. Say which factors are comparable, '
    "that no stated difference in conditions was found, and that the contradiction therefore stands on the stated "
    "evidence.\n"
    '- "insufficient": the claims oppose at face value, but the passages do not state comparable context for both '
    "claims. Say what is missing (which factors only one paper states, or that none is stated by both) and that a "
    "contradiction can therefore be neither established nor ruled out.\n"
    'Return a JSON object: {"explanation": "two to four sentences", "factors_cited": ["names of the table factors '
    'you mention"]}.'
)

SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}, "factors_cited": {"type": "array", "items": {"type": "string"}}},
    "required": ["explanation", "factors_cited"],
}


def _row(f: FactorComparison) -> dict:
    return {"factor": f.factor, "a_value": ", ".join(f.a) or None, "b_value": ", ".join(f.b) or None,
            "a_quote": f.a_quote, "b_quote": f.b_quote, "paper_level": f.inherited}


def apparent_conflict(rel: Relationship, a: Claim, b: Claim) -> str:
    """How the claims oppose at face value: the stance step's restatement if there is one, else the claims."""
    if rel.llm_note:
        return f"{a.source_citation} (A) versus {b.source_citation} (B). {rel.llm_note}"
    return f"{a.source_citation} (A): “{a.claim_text}” versus {b.source_citation} (B): “{b.claim_text}”"


def evidence_record(conflict: ConflictAnalysis, rel: Relationship, a: Claim, b: Claim) -> dict:
    """Everything the explanation may rely on. Built from the decision's own factor table."""
    one_sided = [{"factor": f.factor, "stated_by": "A" if f.a else "B", "value": ", ".join(f.a or f.b),
                  "quote": f.a_quote or f.b_quote} for f in rel.factors if f.status == "unknown" and (f.a or f.b)]
    return {
        "relation": rel.relation.value,
        "conflict_verdict": VERDICT_NAME[conflict.verdict],
        "apparent_conflict": apparent_conflict(rel, a, b),
        "claim_a": {"paper": a.source_citation, "text": a.claim_text},
        "claim_b": {"paper": b.source_citation, "text": b.claim_text},
        "decisive_mismatches": [_row(f) for f in conflict.differing_factors],
        "comparable_factors": [_row(f) for f in conflict.shared_factors],
        "one_sided_factors": one_sided,
    }


def render_evidence(record: dict) -> str:
    def table(rows: list[dict]) -> str:
        if not rows:
            return "  (none)"
        return "\n".join(f"  - {r['factor']}: A = {r['a_value']} [quote: “{r['a_quote'] or '-'}”] | "
                         f"B = {r['b_value']} [quote: “{r['b_quote'] or '-'}”]" for r in rows)

    one_sided = "\n".join(f"  - {r['factor']}: only {r['stated_by']} states it ({r['value']})" for r in record["one_sided_factors"]) or "  (none)"
    return (f"CLAIM A ({record['claim_a']['paper']}): {record['claim_a']['text']}\n"
            f"CLAIM B ({record['claim_b']['paper']}): {record['claim_b']['text']}\n\n"
            f"DECISION: relation = {record['relation']}; conflict verdict = {record['conflict_verdict']}\n\n"
            f"Context factors that DIFFER (stated by both papers):\n{table(record['decisive_mismatches'])}\n"
            f"Context factors that are COMPARABLE (stated by both papers):\n{table(record['comparable_factors'])}\n"
            f"Context factors stated by ONE paper only:\n{one_sided}")


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def check_explanation(out: dict | None, record: dict) -> tuple[str | None, str | None]:
    """(explanation, None) if the model's text passes every grounding check, else (None, reason)."""
    if not out or not isinstance(out.get("explanation"), str):
        return None, "no usable response"
    text = " ".join(out["explanation"].split())
    words = len(text.split())
    if words < 8 or words > MAX_WORDS:
        return None, f"length {words} words"
    rows = record["decisive_mismatches"] + record["comparable_factors"]
    in_table = {r["factor"] for r in rows} | {r["factor"] for r in record["one_sided_factors"]}
    cited = [str(x) for x in (out.get("factors_cited") or [])]
    outside = sorted(set(cited) - in_table)
    if outside:
        return None, f"cites factors that are not in the table: {outside}"
    decisive = {r["factor"] for r in record["decisive_mismatches"]}
    if record["conflict_verdict"] == "context_dependent" and not decisive & set(cited):
        return None, "cites none of the decisive mismatches"
    evidence = " ".join([record["claim_a"]["text"], record["claim_b"]["text"], record["apparent_conflict"],
                         *(str(r.get(k) or "") for r in rows for k in ("a_value", "b_value", "a_quote", "b_quote")),
                         *(f"{r['value']} {r['quote'] or ''}" for r in record["one_sided_factors"])]).lower()
    low = text.lower()
    for factor, words_ in FACTOR_WORDS.items():
        if factor in in_table:
            continue
        hit = next((w for w in words_ if w in low and w not in evidence), None)
        if hit:
            return None, f"names a factor that is not in the table: {factor!r} ('{hit}')"
    if re.search(r"\b(context )?table\b|user message", low):
        return None, "refers to the table instead of the papers"
    if record["conflict_verdict"] == "context_dependent":
        def stated(value: str | None) -> bool:
            v = " ".join(str(value or "").lower().split())
            return bool(v) and (v in low or fuzz.partial_ratio(v, low) >= 90)

        if not any(stated(r["a_value"]) and stated(r["b_value"]) for r in record["decisive_mismatches"] if r["factor"] in cited):
            return None, "states no decisive mismatch with both papers' values as given"
    numbers = {n.replace(",", "") for n in _NUMBER.findall(text)}
    known = {n.replace(",", "") for n in _NUMBER.findall(evidence)}
    invented = sorted(n for n in numbers if n not in known and len(n) > 1)
    if invented:
        return None, f"contains numbers that are not in the evidence: {invented}"
    return text, None


def explain_one(conflict: ConflictAnalysis, rel: Relationship, a: Claim, b: Claim, llm: BudgetedLLM) -> ConflictAnalysis:
    """Attach the apparent conflict and, if it passes the checks, the LLM's wording of the verdict."""
    record = evidence_record(conflict, rel, a, b)
    update: dict = {"apparent_conflict": record["apparent_conflict"]}
    if not llm.enabled:
        return conflict.model_copy(update=update)
    user = render_evidence(record)
    out = llm.complete_json(SYSTEM, user, SCHEMA, max_tokens=350)
    text, problem = check_explanation(out, record) if out is not None else (None, None)
    if out is not None and text is None:
        # one retry: at temperature 0 the same prompt gives the same text, so say what was wrong
        retry = llm.complete_json(SYSTEM, f"{user}\n\nYour previous explanation was rejected: {problem}. Write it again and follow every rule.",
                                  SCHEMA, max_tokens=350)
        if retry is not None:
            text, problem = check_explanation(retry, record)
    if text:
        update.update(llm_explanation=text, explanation_source="llm")
    elif out is not None:
        update["explanation_note"] = f"LLM explanation rejected ({problem}); template explanation shown"
    return conflict.model_copy(update=update)


def explain_conflicts(conflicts: list[ConflictAnalysis], relationships: list[Relationship], claims: dict[str, Claim],
                      llm: BudgetedLLM, *, limit: int | None = None, concurrency: int = 4) -> list[ConflictAnalysis]:
    """Explain the first ``limit`` conflicts (they arrive sorted: genuine, context-dependent, insufficient)."""
    rels = {r.relation_id: r for r in relationships}
    todo = set(range(len(conflicts)) if limit is None else range(min(limit, len(conflicts))))

    def run(k: int) -> ConflictAnalysis:
        c = conflicts[k]
        rel = rels[c.relation_id]
        a, b = claims[c.claim_a_id], claims[c.claim_b_id]
        if k not in todo:
            return c.model_copy(update={"apparent_conflict": apparent_conflict(rel, a, b)})
        try:
            return explain_one(c, rel, a, b, llm)
        except Exception:  # noqa: BLE001 — an explanation must never break an analysis
            log.exception("conflict explanation failed")
            return c

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(run, range(len(conflicts))))


def explanation_artifact(conflict: ConflictAnalysis, rel: Relationship, a: Claim, b: Claim) -> dict:
    """The record written to evaluation artefacts and printed in the terminal."""
    record = evidence_record(conflict, rel, a, b)
    record.update(explanation=conflict.llm_explanation or conflict.explanation, explanation_source=conflict.explanation_source,
                  explanation_note=conflict.explanation_note, template_explanation=conflict.explanation)
    return record
