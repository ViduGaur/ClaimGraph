"""Audit of the explanation layer, from the cheapest evidence to the most independent.

1. **Structural**: every quote in an explanation must occur in the paper it is attributed to (:func:`structural_check`).
2. **Acceptance**: how many LLM wordings passed the grounding checks of :mod:`app.reasoning.explanations`, and why the
   rest were replaced by the deterministic template.
3. **Independent judges**: two models that took no part in the explanation flag statements the evidence record does not
   support, added causal reasons, and conclusions that differ from the decision.

Faithfulness to the evidence record is not correctness: an explanation inherits any error in its stance or context table.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.evaluation.experiment import TopicData
from app.evaluation.model_annotation import ANNOTATORS, Annotator, CallBudget, call_annotator
from app.providers.llm.base import LLMError
from app.reasoning.alignment import claim_texts, quote_grounded
from app.reasoning.explanations import render_evidence

JUDGES: list[Annotator] = [a for a in ANNOTATORS if a.id in ("gemini", "qwen")]

JUDGE_SYSTEM = (
    "You check an explanation against the evidence it was written from. The EVIDENCE is complete: the explanation is "
    "allowed to use nothing else. Do not use outside knowledge and do not judge whether the decision is right.\n"
    "Report:\n"
    "- \"unsupported\": every statement in the explanation that asserts a fact, value, condition or number that is not "
    "in the evidence (an empty list if there is none). Restating or summarising the evidence is fine.\n"
    "- \"adds_causal_reason\": true if the explanation claims WHY the results differ (a mechanism or cause), beyond "
    "saying that the stated conditions differ or are not stated.\n"
    "- \"matches_decision\": true if the explanation's conclusion agrees with the DECISION in the evidence "
    "(context_dependent = not a direct contradiction because stated conditions differ; genuine = the contradiction "
    "stands; insufficient = cannot be established or ruled out).\n"
    "Respond with ONLY a JSON object: {\"unsupported\": [\"...\"], \"adds_causal_reason\": true or false, "
    "\"matches_decision\": true or false}"
)


def parse_judgement(out: dict) -> dict:
    unsupported = out.get("unsupported")
    if not isinstance(unsupported, list) or "matches_decision" not in out:
        raise LLMError("judgement is missing fields")
    return {"unsupported": [str(x)[:300] for x in unsupported], "adds_causal_reason": bool(out.get("adds_causal_reason")),
            "matches_decision": bool(out.get("matches_decision"))}


def collect(datas: dict[str, TopicData]) -> list[dict]:
    rows = []
    for tid in sorted(datas):
        for rec in datas[tid].records:
            if rec.get("explanation"):
                rows.append({"topic": tid, "pair_id": rec["pair_id"], **rec["explanation"]})
    return rows


def structural_check(datas: dict[str, TopicData]) -> dict:
    """Every quote in every explanation record must occur in the paper it is attributed to."""
    quotes = grounded = 0
    for data in datas.values():
        for rec in data.records:
            e = rec.get("explanation")
            if not e:
                continue
            texts = {"a": claim_texts(data.claims[rec["a"]], data.papers.get(data.claims[rec["a"]].paper_id)),
                     "b": claim_texts(data.claims[rec["b"]], data.papers.get(data.claims[rec["b"]].paper_id))}
            for row in e["decisive_mismatches"] + e["comparable_factors"]:
                for side in ("a", "b"):
                    q = row[f"{side}_quote"]
                    if q:
                        quotes += 1
                        grounded += quote_grounded(q, texts[side].local) or quote_grounded(q, texts[side].paper)
    return {"quotes": quotes, "found_in_source": grounded}


def run_judges(rows: list[dict], out_dir: Path, api_key: str, *, concurrency: int = 6) -> dict:
    """Each judge reads each explanation once (resumable). Returns HTTP-call and cost totals."""
    log_path = out_dir / "explanation_judge_calls.jsonl"
    done: set[tuple[str, str]] = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if "matches_decision" in r:
                done.add((r["judge"], r["pair_id"]))
    todo = [(j, r) for j in JUDGES for r in rows if (j.id, r["pair_id"]) not in done]
    budget = CallBudget(int(len(todo) * 1.5) + 5)
    lock = threading.Lock()

    def run(job: tuple[Annotator, dict]) -> None:
        judge, row = job
        user = f"EVIDENCE\n{render_evidence(row)}\n\nEXPLANATION\n{row['explanation']}"
        out = {"judge": judge.id, "model": judge.model, "pair_id": row["pair_id"], "explanation_source": row["explanation_source"]}
        try:
            out.update(call_annotator(judge, JUDGE_SYSTEM, user, api_key, budget=budget, parse=parse_judgement, max_tokens=400))
        except LLMError as exc:
            out.update(error=str(exc), usage=getattr(exc, "usage", None))
        with lock, log_path.open("a") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(run, todo))
    calls = cost = 0
    for line in log_path.read_text().splitlines():
        u = json.loads(line).get("usage") or {}
        calls += u.get("http_calls") or 0
        cost += float(u.get("cost") or 0.0)
    return {"new_requests": len(todo), "http_calls_total": calls, "cost_total": round(cost, 4)}


def summarise(datas: dict[str, TopicData], out_dir: Path) -> dict:
    rows = collect(datas)
    by_pair = {r["pair_id"]: r for r in rows}
    sources = Counter((r["conflict_verdict"], r["explanation_source"]) for r in rows)
    reasons = Counter()
    for r in rows:
        if r.get("explanation_note"):
            note = r["explanation_note"].split("(", 1)[1].rsplit(")", 1)[0]
            reasons[note.split(":")[0].split(" [")[0].strip()[:60]] += 1
    summary: dict = {"explanations": len(rows), "llm_worded": sum(1 for r in rows if r["explanation_source"] == "llm"),
                     "template": sum(1 for r in rows if r["explanation_source"] == "template"),
                     "by_verdict_and_source": {f"{v}/{s}": n for (v, s), n in sorted(sources.items())},
                     "rejection_reasons": dict(reasons.most_common()), "structure": structural_check(datas)}
    judge_path = out_dir / "explanation_judge_calls.jsonl"
    flagged: list[dict] = []
    if judge_path.exists():
        per: dict[str, dict[str, dict]] = {}
        for line in judge_path.read_text().splitlines():
            r = json.loads(line)
            if "matches_decision" in r and r["pair_id"] in by_pair:
                per.setdefault(r["judge"], {})[r["pair_id"]] = r
        judges = {}
        for judge, verdicts in per.items():
            for source in ("llm", "template"):
                vs = [v for pid, v in verdicts.items() if by_pair[pid]["explanation_source"] == source]
                if vs:
                    judges[f"{judge}/{source}"] = {"n": len(vs), "fully_supported": sum(1 for v in vs if not v["unsupported"]),
                                                   "adds_causal_reason": sum(1 for v in vs if v["adds_causal_reason"]),
                                                   "matches_decision": sum(1 for v in vs if v["matches_decision"])}
        summary["judges"] = judges
        ids = sorted(per)
        if len(ids) == 2:
            common = [pid for pid in per[ids[0]] if pid in per[ids[1]] and by_pair[pid]["explanation_source"] == "llm"]
            clean = lambda j, pid: not per[j][pid]["unsupported"] and not per[j][pid]["adds_causal_reason"] and per[j][pid]["matches_decision"]  # noqa: E731
            summary["llm_worded_clean_by_both_judges"] = {"k": sum(1 for pid in common if clean(ids[0], pid) and clean(ids[1], pid)), "n": len(common)}
            summary["llm_worded_flagged_by_both_judges"] = {"k": sum(1 for pid in common if not clean(ids[0], pid) and not clean(ids[1], pid)), "n": len(common)}
            summary["judges_agree"] = {"k": sum(1 for pid in common if clean(ids[0], pid) == clean(ids[1], pid)), "n": len(common)}
            for pid in common:
                if not clean(ids[0], pid) and not clean(ids[1], pid):
                    flagged.append({"pair_id": pid, "explanation": by_pair[pid]["explanation"],
                                    **{j: {k: per[j][pid][k] for k in ("unsupported", "adds_causal_reason", "matches_decision")} for j in ids}})
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "explanations.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "explanation_audit.json").write_text(json.dumps({**summary, "flagged_by_both_judges": flagged}, indent=1, ensure_ascii=False))
    return {**summary, "flagged_by_both_judges": flagged}
