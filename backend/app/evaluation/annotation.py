"""Gold-label workflow: sample pairs → blind annotation sheets → aggregate labels and agreement.

Annotators never see system output. They answer two questions per pair, which mirror the
two things ClaimGraph has to get right:

1. **stance** — at face value, do the claims ``agree``, ``oppose``, only ``partial``-ly bear on
   each other, or are they ``unrelated``?
2. **context** — are the stated conditions ``comparable``, ``different`` in a way that could
   explain the difference, or ``not_stated``?

The gold relation and conflict verdict are derived from the two answers (:func:`derive_label`,
:func:`derive_verdict`), so stance
detection and the context gate can also be scored separately.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.experiment import SYSTEM_SPECS, TopicData, predict_all
from app.evaluation.metrics import ABSTAIN, RELATIONS

STANCES = ["agree", "oppose", "partial", "unrelated"]
CONTEXTS = ["comparable", "different", "not_stated"]
CONFLICT_LABELS = {"CONTRADICTS", "QUALIFIES", ABSTAIN}
AGREE_LABELS = {"SUPPORTS", "EXTENDS"}


def derive_label(stance: str, context: str | None) -> str | None:
    """Two annotation answers → the gold relation (``None`` if incomplete).

    An opposing pair whose context is not stated is a CONTRADICTS at the relation level; that its conditions
    cannot be compared is recorded in the conflict verdict (:func:`derive_verdict`), not as a sixth relation.
    """
    if stance == "unrelated":
        return "UNRELATED"
    if stance not in STANCES or context not in CONTEXTS:
        return None
    if stance == "agree":
        return "EXTENDS" if context == "different" else "SUPPORTS"
    if stance == "partial":
        return "QUALIFIES"
    return "QUALIFIES" if context == "different" else "CONTRADICTS"


def derive_verdict(stance: str, context: str | None) -> str | None:
    """Gold conflict verdict for a face-value conflict (stance ``oppose``), else ``None``."""
    if stance != "oppose" or context not in CONTEXTS:
        return None
    return {"comparable": "genuine", "different": "context_dependent", "not_stated": "insufficient"}[context]


# --------------------------------------------------------------------------- sampling
@dataclass
class Item:
    item_id: str
    pair_id: str
    topic: str
    stratum: str  # one of STRATA
    stratum_size: int = 0  # pairs of this stratum in the topic's candidate pool (for weighting)


STRATA = ["full_assert", "gate_input", "conflict", "agreement", "background"]


def stratum_of(labels: dict[str, str | None]) -> str:
    """Pooling over systems, most specific stratum first.

    * ``full_assert`` — the full system asserts a contradiction or abstains (rare; annotate all of them,
      otherwise its false-contradiction rate cannot be estimated);
    * ``gate_input`` — the full system sees a face-value conflict (what the context gate decides on);
    * ``conflict`` — only some other system calls it a conflict;
    * ``agreement`` — some system calls it agreement, none a conflict;
    * ``background`` — no system relates the pair.
    """
    informative = {name: lab for name, lab in labels.items() if name != "similarity" and lab}
    if labels.get("hybrid_full") in ("CONTRADICTS", ABSTAIN):
        return "full_assert"
    if labels.get("hybrid_no_gate") == "CONTRADICTS":
        return "gate_input"
    if any(lab in CONFLICT_LABELS for lab in informative.values()):
        return "conflict"
    if any(lab in AGREE_LABELS for lab in informative.values()):
        return "agreement"
    return "background"


def sample_items(datas: list[TopicData], quotas: dict[str, int], seed: int = 13, exclude: set[str] | None = None) -> list[Item]:
    """Stratified pooled sample per topic. A negative quota takes the whole stratum; the unused quota of a
    small stratum flows to the next one."""
    items: list[Item] = []
    exclude = exclude or set()
    for data in datas:
        preds = predict_all(data, [s.name for s in SYSTEM_SPECS])
        pools: dict[str, list[str]] = defaultdict(list)
        for rec in data.records:
            pools[stratum_of(preds[rec["pair_id"]])].append(rec["pair_id"])
        rng = random.Random(f"{seed}:{data.topic.id}")
        spare = 0
        chosen: dict[str, list[str]] = {}
        for name in STRATA:
            pool = sorted(p for p in pools[name] if p not in exclude)
            rng.shuffle(pool)
            quota = quotas.get(name, 0)
            if quota < 0:  # take the whole stratum; nothing spills over
                chosen[name] = pool
                continue
            want = quota + spare
            chosen[name] = pool[:want]
            spare = want - len(chosen[name])
        for name, pids in chosen.items():
            for pid in pids:
                iid = "item_" + hashlib.sha1(f"{seed}:{pid}".encode()).hexdigest()[:10]
                items.append(Item(iid, pid, data.topic.id, name, len(pools[name])))
    return items


# --------------------------------------------------------------------------- sheets
def item_payload(item: Item, data: TopicData) -> dict:
    rec = next(r for r in data.records if r["pair_id"] == item.pair_id)
    out = {"item_id": item.item_id, "topic": data.topic.id, "question": data.topic.question}
    for tag, cid in (("a", rec["a"]), ("b", rec["b"])):
        c = data.claims[cid]
        p = data.papers[c.paper_id]
        out[tag] = {"citation": c.source_citation, "title": p.title, "section": c.section, "claim": c.claim_text,
                    "passage": " ".join(e.text for e in c.evidence), "abstract": p.abstract or ""}
    return out


def assign(items: list[Item], annotators: list[str], per_item: int, overlap: int, seed: int = 13) -> dict[str, list[str]]:
    """Each item goes to ``per_item`` annotators (balanced); the first ``overlap`` items go to everyone."""
    rng = random.Random(seed)
    order = [i.item_id for i in items]
    rng.shuffle(order)
    out: dict[str, list[str]] = {a: [] for a in annotators}
    for k, iid in enumerate(order):
        if k < overlap or per_item >= len(annotators):
            who = annotators
        else:
            who = [annotators[(k + j) % len(annotators)] for j in range(per_item)]
        for a in who:
            out[a].append(iid)
    return out


CSV_FIELDS = ["item_id", "topic", "question", "paper_a", "claim_a", "passage_a", "paper_b", "claim_b", "passage_b",
              "stance", "context", "bad_claim", "note"]


def write_sheets(items: list[Item], datas: dict[str, TopicData], annotators: list[str], out_dir: Path, *, per_item: int = 2,
                 overlap: int = 30, seed: int = 13, batch: str = "batch1") -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {i.item_id: item_payload(i, datas[i.topic]) for i in items}
    assignment = assign(items, annotators, per_item, overlap, seed)
    template = (Path(__file__).parent / "annotate_template.html").read_text()
    for annotator, ids in assignment.items():
        rng = random.Random(f"{seed}:{annotator}")
        by_topic: dict[str, list[str]] = defaultdict(list)
        for iid in ids:
            by_topic[payloads[iid]["topic"]].append(iid)
        ordered = []
        for topic in sorted(by_topic):  # one topic at a time (less context switching), random order inside
            rng.shuffle(by_topic[topic])
            ordered += by_topic[topic]
        rows = [payloads[i] for i in ordered]
        with (out_dir / f"{batch}_{annotator}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({"item_id": r["item_id"], "topic": r["topic"], "question": r["question"],
                            "paper_a": f"{r['a']['citation']} — {r['a']['title']}", "claim_a": r["a"]["claim"], "passage_a": r["a"]["passage"],
                            "paper_b": f"{r['b']['citation']} — {r['b']['title']}", "claim_b": r["b"]["claim"], "passage_b": r["b"]["passage"],
                            "stance": "", "context": "", "bad_claim": "", "note": ""})
        html = template.replace("__ANNOTATOR__", annotator).replace("__BATCH__", batch).replace(
            "__ITEMS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
        (out_dir / f"{batch}_{annotator}.html").write_text(html)
    key = {"batch": batch, "seed": seed, "per_item": per_item, "overlap": overlap,
           "items": [i.__dict__ for i in items], "assignment": assignment}
    (out_dir / f"{batch}_key.json").write_text(json.dumps(key, indent=1))
    return {a: len(ids) for a, ids in assignment.items()}


# --------------------------------------------------------------------------- labels
def read_label_file(path: Path) -> dict[str, dict]:
    """A filled sheet (the tool's JSON export, or the CSV) → {item_id: {stance, context, bad_claim, note}}."""
    rows: list[dict]
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        rows = raw["labels"] if isinstance(raw, dict) else raw
    else:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        stance = (r.get("stance") or "").strip().lower()
        context = (r.get("context") or "").strip().lower() or None
        if stance not in STANCES:
            continue
        out[r["item_id"]] = {"stance": stance, "context": context if context in CONTEXTS else None,
                             "bad_claim": str(r.get("bad_claim") or "").strip().lower() in ("1", "true", "yes", "a", "b", "ab", "both"),
                             "note": (r.get("note") or "").strip()}
    return out


def load_labels(labels_dir: Path) -> dict[str, dict[str, dict]]:
    """{item_id: {annotator: answer}} from every ``<anything>_<annotator>.(json|csv)`` in the directory."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(labels_dir.glob("*")):
        if path.suffix not in (".json", ".csv") or path.name.startswith("adjudicated") or path.name.endswith("_key.json"):
            continue
        annotator = path.stem.split("_")[-1]
        for iid, answer in read_label_file(path).items():
            out[iid][annotator] = answer
    return out


def gold_labels(labels: dict[str, dict[str, dict]], adjudicated: dict[str, dict] | None = None) -> tuple[dict[str, dict], list[str]]:
    """Resolve annotations to one gold answer per item.

    Agreement on the derived label → gold. Otherwise the adjudicated answer is used if there is
    one; items still unresolved are returned separately and excluded from scoring.
    """
    gold: dict[str, dict] = {}
    unresolved: list[str] = []
    for iid, answers in labels.items():
        if adjudicated and iid in adjudicated:
            a = adjudicated[iid]
            gold[iid] = {**a, "label": derive_label(a["stance"], a["context"]),
                         "verdict": derive_verdict(a["stance"], a["context"]), "source": "adjudicated"}
            continue
        derived = [derive_label(a["stance"], a["context"]) for a in answers.values()]
        derived = [d for d in derived if d]
        if not derived:
            continue
        top, n = Counter(derived).most_common(1)[0]
        if n == len(derived) or (len(derived) >= 3 and n > len(derived) / 2):
            winner = next(a for a in answers.values() if derive_label(a["stance"], a["context"]) == top)
            gold[iid] = {**winner, "label": top, "verdict": derive_verdict(winner["stance"], winner["context"]),
                         "source": "single" if len(derived) == 1 else "agreement",
                         "bad_claim": any(a["bad_claim"] for a in answers.values())}
        else:
            unresolved.append(iid)
    return gold, unresolved


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement_report(labels: dict[str, dict[str, dict]]) -> dict:
    """Pairwise Cohen's kappa and raw agreement on the stance, the context answer and the derived label."""
    annotators = sorted({a for answers in labels.values() for a in answers})
    report: dict = {"annotators": annotators, "items": len(labels),
                    "multiply_annotated": sum(1 for v in labels.values() if len(v) > 1), "pairs": {}}
    views = {
        "label": lambda a: derive_label(a["stance"], a["context"]),
        "stance": lambda a: a["stance"],
        "context": lambda a: a["context"] if a["stance"] != "unrelated" else None,
    }
    pooled: dict[str, list[tuple[str, str]]] = {k: [] for k in views}
    for i, x in enumerate(annotators):
        for y in annotators[i + 1:]:
            entry = {}
            for name, fn in views.items():
                pairs = []
                for answers in labels.values():
                    if x in answers and y in answers:
                        vx, vy = fn(answers[x]), fn(answers[y])
                        if name == "context":  # only where both judged the claims related
                            if answers[x]["stance"] == "unrelated" or answers[y]["stance"] == "unrelated":
                                continue
                        if vx is not None and vy is not None:
                            pairs.append((vx, vy))
                pooled[name] += pairs
                k = cohen_kappa(pairs)
                entry[name] = {"n": len(pairs), "agreement": round(sum(1 for p, q in pairs if p == q) / len(pairs), 3) if pairs else None,
                               "kappa": round(k, 3) if k is not None else None}
            report["pairs"][f"{x}-{y}"] = entry
    report["pooled"] = {name: {"n": len(p), "agreement": round(sum(1 for a, b in p if a == b) / len(p), 3) if p else None,
                               "kappa": round(cohen_kappa(p), 3) if p else None} for name, p in pooled.items()}
    return report


assert set(RELATIONS) >= {derive_label(s, c) for s in STANCES for c in CONTEXTS}
