"""Query understanding: research question → search queries + key terms."""
from __future__ import annotations

import re

from app.extraction.context import canonical_entities, extract_datasets
from app.providers.llm import BudgetedLLM
from app.schemas.api import QueryPlan

_FRAMES = re.compile(
    r"^(what (are|is) (the )?(current |latest |recent )?(findings|evidence|state of (the )?art|research|consensus) (on|about|regarding|for)|"
    r"what do we know about|how (well )?(does|do|did|can|is|are)|does|do|is|are|can|should|why (does|do|is|are)|"
    r"to what extent (does|do|is|are)|what)\s+", re.I)
_STOP = set("""a an the of on in for to and or with by from as at is are was were be been this that these those it its
their they which when while than can may also such do does did whether there any about into over under versus vs
current findings evidence research recent latest effect effects impact role""".split())
_BIOMED = re.compile(r"\b(patient|clinical|disease|drug|therapy|treatment|cancer|tumou?r|gene|protein|covid|sars|vaccine|"
                     r"diabetes|cardio|mortality|trial|randomi[sz]ed|cohort|mice|placebo|dose|infection|brain|neuro|"
                     r"insulin|glucose|glyc|diet|fasting|calori|obes|metabol|nutrition|blood pressure|hypertens|weight loss|"
                     r"exercise|sleep|depress|anxiety|pregnan|smok|alcohol|vitamin|microbiome|immun|surgery|symptom)\w*", re.I)


def heuristic_plan(question: str) -> QueryPlan:
    q = question.strip().rstrip("?").strip()
    core = _FRAMES.sub("", q)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", core)
    key_terms = [w for w in words if w.lower() not in _STOP and len(w) > 2]
    entities = canonical_entities(question) + [d for d in extract_datasets(question)]
    primary = " ".join(key_terms[:8]) or core
    queries = [primary]
    # a shorter, higher-precision variant: entities + the first nouns
    short = " ".join(dict.fromkeys([*entities[:2], *key_terms[:4]]))
    if short and short.lower() != primary.lower():
        queries.append(short)
    return QueryPlan(question=question, search_queries=queries[:2], key_terms=key_terms[:10], entities=entities)


def is_biomedical(plan: QueryPlan) -> bool:
    return bool(_BIOMED.search(plan.question))


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "key_terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["search_queries", "key_terms"],
}


def understand_query(question: str, llm: BudgetedLLM) -> QueryPlan:
    plan = heuristic_plan(question)
    if not llm.enabled:
        return plan
    out = llm.complete_json(
        "Turn a research question into academic search queries (2 queries, 3-7 keywords each, no boolean operators) "
        "and key technical terms. Do not answer the question.",
        question,
        PLAN_SCHEMA,
        max_tokens=120,
    )
    if not out:
        return plan
    queries = [q.strip() for q in out.get("search_queries", []) if isinstance(q, str) and 3 < len(q.strip()) < 120][:2]
    terms = [t.strip() for t in out.get("key_terms", []) if isinstance(t, str) and t.strip()][:10]
    if not queries:
        return plan
    return QueryPlan(question=question, search_queries=queries, key_terms=terms or plan.key_terms,
                     entities=plan.entities, method="llm")
