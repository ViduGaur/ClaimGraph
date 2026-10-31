"""Blinded model-assisted annotation: evaluation sample → three independent LLM annotators →
consensus → researcher adjudication → gold labels.

What the gold labels are, and are not
-------------------------------------
They are **blinded model-assisted consensus labels with researcher adjudication**: three models
from three different providers label every pair independently and zero-shot; a label all three
agree on is taken as is, every other pair is flagged for the researcher to adjudicate, and a
random part of the unanimous pairs is re-checked by the researcher. They are not expert human
annotations and must not be described as such.

Blinding
--------
An annotator model receives the two claims, their passages, the papers' titles and abstracts and
the instructions. It never receives anything ClaimGraph or a baseline produced: no predicted
relation, no verdict, no context table, no NLI or similarity score, no explanation. The prompt is
built by :func:`annotation_prompt` from the claim texts only, and
``tests/test_annotation_models.py`` checks that no system field can reach it.

Annotation calls are logged to ``<batch>/model_calls.jsonl`` with their token usage and cost. They
do not touch the pipeline's LLM cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.evaluation.annotation import Item, cohen_kappa, item_payload
from app.evaluation.experiment import SYSTEM_SPECS, TopicData, predict_all
from app.evaluation.metrics import ABSTAIN, RELATIONS
from app.providers.llm.base import LLMError, parse_json_object

METHOD_NAME = "blinded model-assisted consensus labels with researcher adjudication"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Annotator:
    id: str
    model: str  # OpenRouter model id
    family: str
    extra: tuple[tuple[str, object], ...] = ()  # request fields specific to this model


# three model families from three providers; none of them is the pipeline's own model
ANNOTATORS: list[Annotator] = [
    Annotator("gemini", "google/gemini-3-flash-preview", "Google Gemini 3 Flash", (("reasoning", {"effort": "low"}),)),
    Annotator("qwen", "qwen/qwen3.7-plus", "Alibaba Qwen 3.7 Plus", (("reasoning", {"enabled": False}),)),
    Annotator("haiku", "anthropic/claude-haiku-4.5", "Anthropic Claude Haiku 4.5"),
]

# OpenRouter list prices in US$ per million tokens (input, output), read from /api/v1/models on 2026-10-06
PRICES_PER_M = {"google/gemini-3-flash-preview": (0.50, 3.00), "qwen/qwen3.7-plus": (0.32, 1.28),
                "anthropic/claude-haiku-4.5": (1.00, 5.00)}

# Zero-shot: definitions only, no worked examples.
ANNOTATION_SYSTEM = (
    "You are annotating pairs of claims for a study of how findings in different scientific papers relate to each "
    "other. You will see two claims, A and B, each taken verbatim from a different paper, together with the passage "
    "it comes from and the title and abstract of its paper. Judge ONLY from the text provided. Do not use outside "
    "knowledge about these papers or the field.\n\n"
    "Assign exactly one relation:\n"
    "- SUPPORTS: the two claims assert the same finding about the same relationship (the same direction of effect), "
    "under the same or comparable conditions.\n"
    "- EXTENDS: the two claims are consistent, and one carries the finding into a different setting (another "
    "dataset, scale, task, population or condition) or builds on it.\n"
    "- CONTRADICTS: the two claims make opposing assertions about the same relationship, and the provided text gives "
    "no difference in conditions that accounts for the opposition, so they cannot both hold under equivalent "
    "conditions.\n"
    "- QUALIFIES: one claim narrows, conditions, limits or contextualizes the other. Any apparent disagreement is "
    "attributable to differences stated in the text (for example dataset, population, task, model or architecture, "
    "training setup, experimental conditions, scale, or evaluation metric), so it is not a direct contradiction "
    "under equivalent conditions.\n"
    "- UNRELATED: the claims are not about the same relationship (a different method or factor, or a different "
    "outcome), even if they share a topic.\n\n"
    "'The same relationship' means the same method, intervention or factor assessed on the same kind of outcome.\n\n"
    "Also answer one yes/no question, \"face_value_opposition\": ignoring every difference in setting or "
    "conditions, do the two claims state opposing conclusions about the same relationship?\n\n"
    "Respond with ONLY a JSON object: {\"label\": \"SUPPORTS|EXTENDS|CONTRADICTS|QUALIFIES|UNRELATED\", "
    "\"face_value_opposition\": true or false, \"justification\": \"at most 30 words, citing only the provided text\"}"
)

# keys of a pair record or prediction that must never appear in an annotation prompt
FORBIDDEN_IN_PROMPT = ("nli_ab", "nli_ba", "similarity", "llm_pair", "llm_direct", "explanation", "conflict_verdict",
                       "hybrid_", "rules_", "context gate", "ClaimGraph")


def annotation_prompt(payload: dict, abstract_chars: int = 1200) -> str:
    """The user message for one pair. ``payload`` comes from ``item_payload``: claims, passages, titles, abstracts."""
    def side(tag: str, s: dict) -> str:
        return (f"PAPER {tag}: {s['citation']} - {s['title']}\n"
                f"Abstract {tag}: {(s['abstract'] or 'not available')[:abstract_chars]}\n"
                f"Passage {tag} (section: {s['section']}): {s['passage']}\n"
                f"CLAIM {tag}: {s['claim']}")

    return side("A", payload["a"]) + "\n\n" + side("B", payload["b"])


# --------------------------------------------------------------------------- sampling
STRATA: list[tuple[str, str]] = [
    ("asserted_contradiction", "the full system asserts CONTRADICTS"),
    ("abstained", "the full system sees a conflict and abstains (insufficient evidence)"),
    ("gate_rejected_nli_contra", "NLI says contradiction; the full system sees the conflict and the gate rejects it"),
    ("gate_rejected", "the full system sees a face-value conflict and the gate rejects it (NLI does not flag it)"),
    ("nli_contra_no_conflict", "NLI says contradiction; the full system sees no face-value conflict"),
    ("direct_contra_no_conflict", "a direct-LLM baseline says contradiction; NLI and the full system do not"),
    ("qualifies_partial", "the full system says QUALIFIES without a face-value conflict"),
    ("extends", "the full system says EXTENDS"),
    ("supports", "the full system says SUPPORTS"),
    ("others_relate", "the full system says UNRELATED; another system asserts a relation"),
    ("all_unrelated", "no system relates the pair"),
]
DEFAULT_QUOTAS = {"asserted_contradiction": -1, "abstained": -1, "gate_rejected_nli_contra": -1, "gate_rejected": 30,
                  "nli_contra_no_conflict": 25, "direct_contra_no_conflict": 23, "qualifies_partial": -1, "extends": 15,
                  "supports": 18, "others_relate": 22, "all_unrelated": 8}


def stratum_of(labels: dict[str, str | None]) -> str:
    """Sampling stratum of a pair from the frozen systems' outputs (first matching rule wins)."""
    full, no_gate, nli = labels.get("hybrid_full"), labels.get("hybrid_no_gate"), labels.get("nli")
    direct_contra = "CONTRADICTS" in (labels.get("llm_direct_claims"), labels.get("llm_direct_context"))
    if full == "CONTRADICTS":
        return "asserted_contradiction"
    if full == ABSTAIN:
        return "abstained"
    if no_gate == "CONTRADICTS":
        return "gate_rejected_nli_contra" if nli == "CONTRADICTS" else "gate_rejected"
    if nli == "CONTRADICTS":
        return "nli_contra_no_conflict"
    if direct_contra:
        return "direct_contra_no_conflict"
    if full in ("QUALIFIES", "EXTENDS", "SUPPORTS"):
        return {"QUALIFIES": "qualifies_partial", "EXTENDS": "extends", "SUPPORTS": "supports"}[full]
    others = {labels.get(k) for k in ("llm_direct_claims", "llm_direct_context", "rules_full")}
    return "others_relate" if others - {"UNRELATED", None} else "all_unrelated"


def sample_eval_set(datas: list[TopicData], quotas: dict[str, int] | None = None, seed: int = 2026) -> tuple[list[Item], dict]:
    """Stratified random sample over all topics. A negative quota takes the whole stratum.

    Returns the items and, per stratum, the pool size and the number sampled (for design weights).
    """
    quotas = quotas or DEFAULT_QUOTAS
    names = [s.name for s in SYSTEM_SPECS]
    pools: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for data in sorted(datas, key=lambda d: d.topic.id):
        preds = predict_all(data, names)
        for rec in data.records:
            pools[stratum_of(preds[rec["pair_id"]])].append((data.topic.id, rec["pair_id"]))
    rng = random.Random(seed)
    items: list[Item] = []
    design: dict = {}
    for name, _ in STRATA:
        pool = sorted(pools[name])
        rng.shuffle(pool)
        quota = quotas.get(name, 0)
        chosen = pool if quota < 0 else pool[:quota]
        design[name] = {"pool": len(pool), "sampled": len(chosen)}
        for topic, pid in chosen:
            iid = "ev_" + hashlib.sha1(f"{seed}:{pid}".encode()).hexdigest()[:10]
            items.append(Item(iid, pid, topic, name, len(pool)))
    return items, design


# --------------------------------------------------------------------------- annotator calls
class CallBudget:
    """Hard cap on annotation calls, so a bug cannot spend the key."""

    def __init__(self, max_calls: int):
        self.max_calls, self.calls, self._lock = max_calls, 0, threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.calls >= self.max_calls:
                return False
            self.calls += 1
            return True


def parse_annotation(out: dict) -> dict:
    label = str(out.get("label", "")).upper().strip()
    if label not in RELATIONS:
        raise LLMError(f"label {label!r} is not one of the five relations")
    return {"label": label, "face_value_opposition": bool(out.get("face_value_opposition")),
            "justification": str(out.get("justification", "")).strip()[:400]}


def call_annotator(annotator: Annotator, system: str, user: str, api_key: str, *, max_tokens: int = 500,
                   retries: int = 3, timeout: float = 120.0, budget: CallBudget | None = None, parse=parse_annotation) -> dict:
    """One independent request (fresh context). Returns the parsed answer plus usage; raises LLMError on failure."""
    payload = {"model": annotator.model, "temperature": 0, "max_tokens": max_tokens, "usage": {"include": True},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **dict(annotator.extra)}
    last = "no attempt"
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "http_calls": 0}
    delay = 2.0
    for attempt in range(retries + 1):
        if budget is not None and not budget.take():
            raise LLMError("annotation call budget exhausted")
        t0 = time.time()
        try:
            r = httpx.post(f"{OPENROUTER_URL}/chat/completions", json=payload, timeout=timeout,
                           headers={"Authorization": f"Bearer {api_key}", "X-Title": "ClaimGraph annotation"})
            usage_total["http_calls"] += 1
            body = r.json()
            if r.status_code >= 400 or "choices" not in body:
                last = f"HTTP {r.status_code}: {json.dumps(body)[:200]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break  # not worth retrying
            else:
                usage = body.get("usage") or {}
                usage_total["prompt_tokens"] += usage.get("prompt_tokens") or 0
                usage_total["completion_tokens"] += usage.get("completion_tokens") or 0
                usage_total["cost"] += float(usage.get("cost") or 0.0)
                content = body["choices"][0]["message"].get("content") or ""
                try:
                    return {**parse(parse_json_object(content)), "raw": content, "usage": usage_total,
                            "upstream": body.get("provider"), "seconds": round(time.time() - t0, 2)}
                except LLMError as exc:
                    last = f"unusable answer: {exc}"
                    if len(payload["messages"]) == 2:
                        payload["messages"].append({"role": "user", "content": "Reminder: respond with ONLY the JSON object described."})
        except (httpx.HTTPError, ValueError) as exc:
            last = f"{exc.__class__.__name__}: {exc}"
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 20)
    err = LLMError(f"{annotator.id} failed: {last}")
    err.usage = usage_total  # type: ignore[attr-defined]
    raise err


def run_annotation(payloads: dict[str, dict], out_dir: Path, api_key: str, *, annotators: list[Annotator] | None = None,
                   concurrency: int = 6, max_calls: int = 900, progress=None) -> dict:
    """Label every item with every annotator, independently. Resumable: finished (annotator, item) pairs are kept."""
    annotators = annotators or ANNOTATORS
    log_path = out_dir / "model_calls.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("label"):
                done[(row["annotator"], row["item_id"])] = row
    todo = [(a, iid) for a in annotators for iid in sorted(payloads) if (a.id, iid) not in done]
    budget = CallBudget(max_calls)
    lock = threading.Lock()
    failures: list[str] = []

    def run(job: tuple[Annotator, str]) -> None:
        a, iid = job
        user = annotation_prompt(payloads[iid])
        row = {"annotator": a.id, "model": a.model, "item_id": iid,
               "prompt_sha": hashlib.sha256((ANNOTATION_SYSTEM + "\n" + user).encode()).hexdigest()[:16]}
        try:
            row.update(call_annotator(a, ANNOTATION_SYSTEM, user, api_key, budget=budget))
        except LLMError as exc:
            row.update(label=None, error=str(exc), usage=getattr(exc, "usage", None))
            failures.append(f"{a.id}/{iid}: {exc}")
        with lock:
            with log_path.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row.get("label"):
                done[(a.id, iid)] = row
            if progress and (len(done) % 50 == 0):
                progress(f"{len(done)} labels")

    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(run, todo))
    return {"requested": len(todo), "labels": len(done), "http_calls": budget.calls, "failures": failures}


def usage_summary(out_dir: Path) -> dict:
    """Calls, tokens and cost per annotator, from the call log (failed attempts included)."""
    per: dict[str, dict] = defaultdict(lambda: {"labels": 0, "failed": 0, "http_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
    path = out_dir / "model_calls.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        p = per[row["annotator"]]
        p["labels" if row.get("label") else "failed"] += 1
        u = row.get("usage") or {}
        for k in ("http_calls", "prompt_tokens", "completion_tokens"):
            p[k] += u.get(k) or 0
        p["cost"] += float(u.get("cost") or 0.0)
        p["model"] = row["model"]
    total = {k: sum(p[k] for p in per.values()) for k in ("labels", "failed", "http_calls", "prompt_tokens", "completion_tokens", "cost")}
    return {"per_annotator": dict(per), "total": total}


# --------------------------------------------------------------------------- consensus and gold
def read_model_labels(out_dir: Path) -> dict[str, dict[str, dict]]:
    """{item_id: {annotator: answer}} — the last successful answer of each annotator for each item."""
    labels: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in (out_dir / "model_calls.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("label"):
            labels[row["item_id"]][row["annotator"]] = {k: row[k] for k in ("label", "face_value_opposition", "justification", "model")}
    return labels


def read_adjudication(out_dir: Path) -> dict[str, dict]:
    """Researcher decisions: ``adjudicated.json`` (tool export) or ``adjudicated.csv`` (item_id, final_label, note)."""
    out: dict[str, dict] = {}
    for path in (out_dir / "adjudicated.json", out_dir / "adjudicated.csv"):
        if not path.exists():
            continue
        if path.suffix == ".json":
            raw = json.loads(path.read_text())
            rows = raw["labels"] if isinstance(raw, dict) else raw
        else:
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        for r in rows:
            label = str(r.get("final_label") or r.get("label") or "").upper().strip()
            if label in RELATIONS:
                fvo = r.get("face_value_opposition")
                out[r["item_id"]] = {"label": label, "note": (r.get("note") or "").strip(),
                                     "face_value_opposition": None if fvo in (None, "") else str(fvo).lower() in ("1", "true", "yes")}
    return out


def build_gold(out_dir: Path, items: list[dict], annotator_ids: list[str] | None = None) -> dict:
    """Consensus, disagreement flags, adjudication and the final label for every item."""
    annotator_ids = annotator_ids or [a.id for a in ANNOTATORS]
    labels = read_model_labels(out_dir)
    adjudicated = read_adjudication(out_dir)
    gold: dict[str, dict] = {}
    for it in items:
        iid = it["item_id"]
        answers = labels.get(iid, {})
        votes = [answers[a]["label"] for a in annotator_ids if a in answers]
        if len(votes) < len(annotator_ids):
            agreement, consensus = "incomplete", None
        else:
            top, n = Counter(votes).most_common(1)[0]
            agreement = "unanimous" if n == len(votes) else ("majority" if n > len(votes) / 2 else "split")
            consensus = top if agreement != "split" else None
        fvo_votes = [answers[a]["face_value_opposition"] for a in annotator_ids if a in answers]
        fvo = sum(fvo_votes) > len(fvo_votes) / 2 if fvo_votes else None
        adj = adjudicated.get(iid)
        if adj:
            final, source = adj["label"], "adjudicated"
            if adj["face_value_opposition"] is not None:
                fvo = adj["face_value_opposition"]
        elif agreement == "unanimous":
            final, source = consensus, "unanimous"
        elif agreement == "majority":
            final, source = consensus, "majority (pending adjudication)"
        else:
            final, source = None, "unresolved"
        verdict = None
        if final == "CONTRADICTS":
            verdict = "genuine"
        elif final == "QUALIFIES" and fvo:
            verdict = "context_dependent"
        gold[iid] = {"model_labels": {a: answers[a]["label"] for a in annotator_ids if a in answers},
                     "model_justifications": {a: answers[a]["justification"] for a in annotator_ids if a in answers},
                     "model_face_value_opposition": {a: answers[a]["face_value_opposition"] for a in annotator_ids if a in answers},
                     "consensus": consensus, "agreement": agreement, "flagged": agreement != "unanimous",
                     "adjudicated": adj["label"] if adj else None, "adjudication_note": adj["note"] if adj else None,
                     "final": final, "final_source": source, "face_value_conflict": fvo, "conflict_verdict": verdict}
    doc = {"method": METHOD_NAME, "annotators": [{"id": a.id, "model": a.model, "family": a.family} for a in ANNOTATORS
                                               if a.id in annotator_ids], "items": gold}
    (out_dir / "gold.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return doc


def fleiss_kappa(rows: list[list[str]], categories: list[str]) -> float | None:
    """Fleiss' kappa for items each labelled by the same number of raters."""
    rows = [r for r in rows if r]
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    n, k = len(rows), len(rows[0])
    p_j = [sum(r.count(c) for r in rows) / (n * k) for c in categories]
    p_i = [(sum(r.count(c) ** 2 for c in categories) - k) / (k * (k - 1)) for r in rows]
    p_bar, p_e = sum(p_i) / n, sum(p * p for p in p_j)
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def agreement_report(doc: dict) -> dict:
    ids = [a["id"] for a in doc["annotators"]]
    items = doc["items"]
    complete = [v for v in items.values() if len(v["model_labels"]) == len(ids)]
    report: dict = {"method": doc["method"], "items": len(items), "complete": len(complete),
                    "agreement_levels": dict(Counter(v["agreement"] for v in items.values())),
                    "final_sources": dict(Counter(v["final_source"] for v in items.values())),
                    "label_distribution": {a: dict(Counter(v["model_labels"].get(a) for v in complete)) for a in ids},
                    "final_distribution": dict(Counter(v["final"] for v in items.values())),
                    "fleiss_kappa": None, "pairwise": {}}
    k = fleiss_kappa([[v["model_labels"][a] for a in ids] for v in complete], RELATIONS)
    report["fleiss_kappa"] = round(k, 3) if k is not None else None
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            pairs = [(v["model_labels"][x], v["model_labels"][y]) for v in complete]
            kk = cohen_kappa(pairs)
            report["pairwise"][f"{x}-{y}"] = {"agreement": round(sum(1 for p, q in pairs if p == q) / len(pairs), 3) if pairs else None,
                                              "kappa": round(kk, 3) if kk is not None else None}
    spot = [v for v in items.values() if v["agreement"] == "unanimous" and v["adjudicated"]]
    report["spot_check"] = {"checked": len(spot), "researcher_agrees": sum(1 for v in spot if v["adjudicated"] == v["consensus"])}
    flagged = [v for v in items.values() if v["flagged"]]
    report["adjudication"] = {"flagged": len(flagged), "adjudicated": sum(1 for v in flagged if v["adjudicated"]),
                              "researcher_kept_majority": sum(1 for v in flagged if v["adjudicated"] and v["adjudicated"] == v["consensus"])}
    return report


def write_adjudication_files(out_dir: Path, items: list[dict], payloads: dict[str, dict], doc: dict, *, spot_checks: int = 20,
                             seed: int = 7, batch: str = "eval") -> dict:
    """The researcher's worklist: every flagged pair, plus a random sample of unanimous pairs to re-check."""
    gold = doc["items"]
    flagged = [i["item_id"] for i in items if gold[i["item_id"]]["flagged"]]
    unanimous = sorted(i["item_id"] for i in items if gold[i["item_id"]]["agreement"] == "unanimous")
    rng = random.Random(seed)
    spot = sorted(rng.sample(unanimous, min(spot_checks, len(unanimous))))
    worklist = flagged + spot
    rng.shuffle(worklist)  # the researcher cannot tell a spot check from a disagreement before revealing the votes
    rows = []
    for iid in worklist:
        p, g = payloads[iid], gold[iid]
        rows.append({"item_id": iid, "topic": p["topic"], "question": p["question"], "a": p["a"], "b": p["b"],
                     "kind": "spot_check" if iid in spot else "disagreement",
                     "votes": [{"annotator": a, "label": g["model_labels"][a], "justification": g["model_justifications"][a]}
                               for a in g["model_labels"]]})
    with (out_dir / "to_adjudicate.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        # no 'kind' column: it would tell the researcher which pairs are spot checks before they decide
        w.writerow(["item_id", "topic", "paper_a", "claim_a", "passage_a", "paper_b", "claim_b", "passage_b",
                    "model_votes", "final_label", "note"])
        for r in rows:
            votes = " | ".join(f"{v['annotator']}: {v['label']} ({v['justification']})" for v in r["votes"])
            w.writerow([r["item_id"], r["topic"], f"{r['a']['citation']} — {r['a']['title']}", r["a"]["claim"], r["a"]["passage"],
                        f"{r['b']['citation']} — {r['b']['title']}", r["b"]["claim"], r["b"]["passage"], votes, "", ""])
    template = (Path(__file__).parent / "adjudicate_template.html").read_text()
    html = template.replace("__BATCH__", batch).replace("__ITEMS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    (out_dir / "adjudicate.html").write_text(html)
    return {"flagged": len(flagged), "spot_checks": len(spot), "worklist": len(worklist)}


def payloads_for(items: list[Item], datas: dict[str, TopicData]) -> dict[str, dict]:
    return {i.item_id: item_payload(i, datas[i.topic]) for i in items}
