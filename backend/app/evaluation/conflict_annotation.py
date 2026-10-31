"""Blinded four-state conflict annotation (protocol v2).

Unlike the five-way protocol of ``model_annotation``, this one has a "cannot tell" outcome. For each pair the
annotators judge, separately:

A. face-value stance          oppose / not_oppose / unclear
B. evidence sufficiency       sufficient / insufficient
C. context comparability      comparable / context_dependent        (only when A = oppose and B = sufficient)
D. conflict status            NONE / GENUINE / CONTEXT_DEPENDENT / INSUFFICIENT_EVIDENCE
E. explanatory factors        which of the twelve factors account for a context-dependent judgement

Blinding is as before: three annotator models, zero-shot, one fresh request each, seeing only the two claims, their
passages, titles and abstracts. The labels are blinded model-assisted consensus labels, not human labels.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.evaluation.annotation import Item, cohen_kappa, item_payload
from app.evaluation.experiment import Systems, TopicData, predict_full
from app.evaluation.model_annotation import ANNOTATORS, Annotator, CallBudget, annotation_prompt, call_annotator, fleiss_kappa
from app.providers.llm.base import LLMError
from app.reasoning.alignment import FACTOR_GUIDE

PROTOCOL = "four-state conflict annotation v2"
STANCES = ["oppose", "not_oppose", "unclear"]
SUFFICIENCY = ["sufficient", "insufficient"]
COMPARABILITY = ["comparable", "context_dependent", "not_applicable"]
STATUSES = ["NONE", "GENUINE", "CONTEXT_DEPENDENT", "INSUFFICIENT_EVIDENCE"]
FACTOR_NAMES = [name for name, _ in FACTOR_GUIDE]

# Zero-shot: definitions only, no worked examples.
SYSTEM = (
    "You are annotating pairs of claims for a study of how findings in different scientific papers relate to each "
    "other. You will see two claims, A and B, each taken verbatim from a different paper, together with the passage "
    "it comes from and the title and abstract of its paper. Judge ONLY from the text provided. Do not use outside "
    "knowledge about these papers or the field.\n\n"
    "Answer the following in order. Answer A, B and C each on its own merits; do not choose them to fit D.\n\n"
    "A. FACE-VALUE STANCE. Ignoring every difference in setting or conditions, do the two claims state opposing "
    "conclusions about the same relationship (the same method, intervention or factor assessed on the same kind of "
    "outcome)?\n"
    "- oppose: yes.\n"
    "- not_oppose: no - they agree, or one adds to the other without opposing it, or they are about different "
    "relationships.\n"
    "- unclear: you cannot tell from the text.\n\n"
    "B. EVIDENCE SUFFICIENCY. Do the provided claims and passages contain enough information to determine whether "
    "an opposition between these two claims constitutes a scientific contradiction?\n"
    "- sufficient: the text says enough about how and under what conditions each claim was established to make that "
    "determination.\n"
    "- insufficient: it does not; from this text alone one cannot tell whether an opposition would be a real "
    "contradiction.\n"
    "Answer B for every pair, whatever you answered in A.\n\n"
    "C. CONTEXT COMPARABILITY. Answer only if A = oppose and B = sufficient; otherwise answer not_applicable.\n"
    "- comparable: the two claims were established under conditions comparable enough that the opposition is a "
    "direct scientific contradiction.\n"
    "- context_dependent: the text states a difference between the two settings that plausibly accounts for the "
    "opposition, so both claims could hold, each in its own setting.\n\n"
    "D. FINAL CONFLICT STATUS. Your overall judgement of the pair:\n"
    "- NONE: the claims do not actually oppose each other.\n"
    "- GENUINE: the claims oppose, the text is sufficient to judge, and the settings are comparable: a genuine "
    "scientific contradiction.\n"
    "- CONTEXT_DEPENDENT: the claims oppose, and the text is sufficient to identify a meaningful difference in "
    "context that explains the opposition.\n"
    "- INSUFFICIENT_EVIDENCE: the claims oppose, but the text does not contain enough to determine whether the "
    "opposition is a genuine contradiction. This is a statement about the evidence, not a weaker form of "
    "CONTEXT_DEPENDENT.\n\n"
    "E. EXPLANATORY FACTORS. If you judged the pair context-dependent, list every factor below whose difference "
    "between the two claims accounts for the opposition; otherwise give an empty list. Use these exact names:\n"
    + "\n".join(f"- {name}: {gloss}" for name, gloss in FACTOR_GUIDE) + "\n\n"
    "Respond with ONLY a JSON object: {\"stance\": \"oppose|not_oppose|unclear\", "
    "\"sufficiency\": \"sufficient|insufficient\", "
    "\"comparability\": \"comparable|context_dependent|not_applicable\", "
    "\"status\": \"NONE|GENUINE|CONTEXT_DEPENDENT|INSUFFICIENT_EVIDENCE\", "
    "\"factors\": [factor names], \"justification\": \"at most 35 words, citing only the provided text\"}"
)


def parse(out: dict) -> dict:
    def pick(key: str, allowed: list[str], upper: bool = False) -> str:
        v = str(out.get(key, "")).strip().replace("-", "_").replace(" ", "_")
        v = v.upper() if upper else v.lower()
        if v not in allowed:
            raise LLMError(f"{key} {v!r} is not one of {allowed}")
        return v

    factors = out.get("factors") or []
    if not isinstance(factors, list):
        raise LLMError("factors is not a list")
    known = {n.lower(): n for n in FACTOR_NAMES}
    return {"stance": pick("stance", STANCES), "sufficiency": pick("sufficiency", SUFFICIENCY),
            "comparability": pick("comparability", COMPARABILITY), "status": pick("status", STATUSES, upper=True),
            "factors": sorted({known[str(x).strip().lower()] for x in factors if str(x).strip().lower() in known}, key=FACTOR_NAMES.index),
            "justification": str(out.get("justification", "")).strip()[:400]}


def derived_status(label: dict) -> str | None:
    """The status that answers A-C imply (None when they do not determine one)."""
    if label["stance"] != "oppose":
        return "NONE" if label["stance"] == "not_oppose" else None
    if label["sufficiency"] == "insufficient":
        return "INSUFFICIENT_EVIDENCE"
    return {"comparable": "GENUINE", "context_dependent": "CONTEXT_DEPENDENT"}.get(label["comparability"])


# --------------------------------------------------------------------------------------------- sample
STRATA = {"system_opposed": "the system reads the pair as a face-value conflict (all such pairs, every topic)",
          "prior_opposition_signal": "not system-opposed, but an annotator model saw opposition under the first protocol (all such pairs of batch eval2)",
          "no_signal": "not system-opposed and no annotator model saw opposition under the first protocol (random sample of batch eval2)"}


def sample(datas: dict[tuple[str, str], TopicData], prior_gold: dict, prior_key: list[dict], *, n_no_signal: int = 30,
           seed: int = 2027) -> tuple[list[dict], dict]:
    """Items to annotate. ``datas`` is keyed by (run, topic). Uses system outputs only to choose pairs, never in a prompt."""
    items: list[dict] = []
    seen: set[str] = set()

    def add(run: str, topic: str, pair_id: str, stratum: str) -> None:
        if pair_id not in seen:
            seen.add(pair_id)
            items.append({"item_id": "c4_" + hashlib.sha1(f"{seed}:{pair_id}".encode()).hexdigest()[:10], "pair_id": pair_id,
                          "topic": topic, "run": run, "stratum": stratum})

    for (run, topic), data in sorted(datas.items()):
        systems = Systems(data)
        for rec in data.records:
            if predict_full(systems, "hybrid_no_gate", rec)[0] == "CONTRADICTS":
                add(run, topic, rec["pair_id"], "system_opposed")
    rest = [k for k in prior_key if k["pair_id"] not in seen]
    signal = [k for k in rest if any(prior_gold[k["item_id"]]["model_face_value_opposition"].values())
              or "CONTRADICTS" in prior_gold[k["item_id"]]["model_labels"].values()]
    quiet = sorted((k for k in rest if k not in signal), key=lambda k: k["pair_id"])
    random.Random(seed).shuffle(quiet)
    for k in signal:
        add("main", k["topic"], k["pair_id"], "prior_opposition_signal")
    for k in quiet[:n_no_signal]:
        add("main", k["topic"], k["pair_id"], "no_signal")
    design = {"system_opposed": {"pool": sum(i["stratum"] == "system_opposed" for i in items), "sampled": sum(i["stratum"] == "system_opposed" for i in items)},
              "prior_opposition_signal": {"pool": len(signal), "sampled": len(signal)},
              "no_signal": {"pool": len(quiet), "sampled": min(n_no_signal, len(quiet))}}
    return items, design


def payloads(items: list[dict], datas: dict[tuple[str, str], TopicData]) -> dict[str, dict]:
    return {i["item_id"]: item_payload(Item(i["item_id"], i["pair_id"], i["topic"], i["stratum"]), datas[(i["run"], i["topic"])]) for i in items}


# ---------------------------------------------------------------------------------------------- calls
def read_calls(out_dir: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    path = out_dir / "model_calls.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row.get("status"):
            done[(row["annotator"], row["item_id"])] = row
    return done


def spent(out_dir: Path) -> dict:
    per: dict[str, dict] = defaultdict(lambda: {"labels": 0, "failed": 0, "http_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
    path = out_dir / "model_calls.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        p = per[row["annotator"]]
        p["labels" if row.get("status") else "failed"] += 1
        u = row.get("usage") or {}
        for k in ("http_calls", "prompt_tokens", "completion_tokens"):
            p[k] += u.get(k) or 0
        p["cost"] += float(u.get("cost") or 0.0)
    total = {k: sum(p[k] for p in per.values()) for k in ("labels", "failed", "http_calls", "prompt_tokens", "completion_tokens", "cost")}
    return {"per_annotator": {k: dict(v) for k, v in per.items()}, "total": total}


def run(payload_by_item: dict[str, dict], out_dir: Path, api_key: str, *, annotators: list[Annotator] | None = None,
        concurrency: int = 6, max_calls: int = 900, max_cost: float = 1.0) -> dict:
    """Every item x every annotator, independently. Resumable. Stops issuing calls once ``max_cost`` US$ is logged."""
    annotators = annotators or ANNOTATORS
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "model_calls.jsonl"
    done = read_calls(out_dir)
    todo = [(a, iid) for iid in sorted(payload_by_item) for a in annotators if (a.id, iid) not in done]
    budget, lock = CallBudget(max_calls), threading.Lock()
    state = {"cost": spent(out_dir)["total"]["cost"], "skipped": 0}
    failures: list[str] = []

    def one(job: tuple[Annotator, str]) -> None:
        a, iid = job
        with lock:
            if state["cost"] >= max_cost:
                state["skipped"] += 1
                return
        user = annotation_prompt(payload_by_item[iid])
        row = {"annotator": a.id, "model": a.model, "item_id": iid, "protocol": PROTOCOL,
               "prompt_sha": hashlib.sha256((SYSTEM + "\n" + user).encode()).hexdigest()[:16]}
        try:
            row.update(call_annotator(a, SYSTEM, user, api_key, budget=budget, parse=parse, max_tokens=600))
        except LLMError as exc:
            row.update(status=None, error=str(exc), usage=getattr(exc, "usage", None))
            failures.append(f"{a.id}/{iid}: {exc}")
        with lock:
            state["cost"] += float((row.get("usage") or {}).get("cost") or 0.0)
            with log_path.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one, todo))
    return {"requested": len(todo), "labels": len(read_calls(out_dir)), "http_calls": budget.calls, "skipped_for_cost": state["skipped"],
            "failures": failures, "cost_logged": round(state["cost"], 4)}


# ------------------------------------------------------------------------------------------ reference
def majority(values: list[str]) -> tuple[str | None, str]:
    """(majority value or None, 'unanimous' | 'majority' | 'split')."""
    top, n = Counter(values).most_common(1)[0]
    if n == len(values):
        return top, "unanimous"
    return (top, "majority") if n > len(values) / 2 else (None, "split")


def build_reference(out_dir: Path, items: list[dict], annotator_ids: list[str] | None = None) -> dict:
    """Per-dimension consensus of the annotator models. Pairs without all annotators are left out."""
    annotator_ids = annotator_ids or [a.id for a in ANNOTATORS]
    calls = read_calls(out_dir)
    doc: dict = {"method": "blinded model-assisted consensus labels (four-state protocol v2), unadjudicated",
                 "protocol": PROTOCOL, "annotators": [{"id": a.id, "model": a.model, "family": a.family} for a in ANNOTATORS], "items": {}}
    for it in items:
        labels = {a: calls.get((a, it["item_id"])) for a in annotator_ids}
        if not all(labels.values()):
            continue
        row: dict = {"pair_id": it["pair_id"], "topic": it["topic"], "run": it["run"], "stratum": it["stratum"],
                     "labels": {a: {k: l[k] for k in ("stance", "sufficiency", "comparability", "status", "factors", "justification")} for a, l in labels.items()}}
        for dim in ("stance", "sufficiency", "status"):
            row[dim], row[f"{dim}_agreement"] = majority([l[dim] for l in labels.values()])
        comp = [l["comparability"] for l in labels.values() if l["comparability"] != "not_applicable"]
        top = Counter(comp).most_common(1)[0] if comp else (None, 0)
        row["comparability"] = top[0] if top[1] >= 2 else None  # at least two annotators answered and agree
        derived = [derived_status(l) for l in labels.values()]
        row["derived"] = [d for d in derived]
        row["status_derived"] = majority([d or "UNDETERMINED" for d in derived])[0]
        if row["status_derived"] == "UNDETERMINED":
            row["status_derived"] = None
        votes = Counter(f for l in labels.values() for f in l["factors"])
        row["factors"] = [f for f in FACTOR_NAMES if votes[f] >= 2]
        row["factor_votes"] = dict(votes)
        doc["items"][it["item_id"]] = row
    return doc


def agreement(doc: dict) -> dict:
    ids = [a["id"] for a in doc["annotators"]]
    rows = list(doc["items"].values())
    out: dict = {"pairs": len(rows)}

    def kappas(name: str, cats: list[str], getter, subset=None) -> None:
        sel = [r for r in rows if subset is None or subset(r)]
        table = [[getter(r["labels"][a]) for a in ids] for r in sel]
        pairwise = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pr = [(t[i], t[ids.index(b)]) for t in table]
                k = cohen_kappa(pr) if pr else None
                pairwise[f"{a}-{b}"] = {"kappa": None if k is None else round(k, 3), "agree": round(sum(x == y for x, y in pr) / len(pr), 3) if pr else None}
        fk = fleiss_kappa(table, cats) if table else None
        out[name] = {"n": len(sel), "fleiss_kappa": None if fk is None else round(fk, 3),
                     "unanimous": sum(len(set(t)) == 1 for t in table), "majority": sum(len(set(t)) == 2 for t in table),
                     "split": sum(len(set(t)) == 3 for t in table), "pairwise": pairwise,
                     "per_annotator": {a: dict(Counter(t[i] for t in table)) for i, a in enumerate(ids)}}

    kappas("stance", STANCES, lambda l: l["stance"])
    kappas("sufficiency", SUFFICIENCY, lambda l: l["sufficiency"])
    kappas("sufficiency_given_majority_oppose", SUFFICIENCY, lambda l: l["sufficiency"], lambda r: r["stance"] == "oppose")
    kappas("comparability_all_three_answered", ["comparable", "context_dependent"], lambda l: l["comparability"],
           lambda r: all(l["comparability"] != "not_applicable" for l in r["labels"].values()))
    kappas("status", STATUSES, lambda l: l["status"])
    kappas("status_given_majority_oppose", STATUSES, lambda l: l["status"], lambda r: r["stance"] == "oppose")
    # does the final status follow from A-C for each annotator?
    out["status_matches_derived"] = {a: {"n": len(rows), "same": sum(derived_status(r["labels"][a]) == r["labels"][a]["status"] for r in rows),
                                         "undetermined": sum(derived_status(r["labels"][a]) is None for r in rows)} for a in ids}
    # factors: among pairs with status CONTEXT_DEPENDENT for at least two annotators
    ctx = [r for r in rows if sum(l["status"] == "CONTEXT_DEPENDENT" for l in r["labels"].values()) >= 2]
    per_factor = {}
    for fname in FACTOR_NAMES:
        table = [["y" if fname in r["labels"][a]["factors"] else "n" for a in ids] for r in ctx]
        k = fleiss_kappa(table, ["y", "n"]) if table else None
        per_factor[fname] = {"named_by_any": sum("y" in t for t in table), "named_by_two": sum(t.count("y") >= 2 for t in table),
                             "fleiss_kappa": None if k is None else round(k, 3)}
    jac = []
    for r in ctx:
        sets = [set(r["labels"][a]["factors"]) for a in ids]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                jac.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    out["factors"] = {"pairs": len(ctx), "mean_pairwise_jaccard": round(sum(jac) / len(jac), 3) if jac else None, "per_factor": per_factor}
    return out
