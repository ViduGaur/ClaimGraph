"""Per-claim frame extraction for the C4 pairs (experiment E3): one cached LLM call per distinct claim.

The model sees one claim with its passage and abstract. It never sees the other claim of a pair, a system
decision or a reference label. Raw outputs are stored in ``eval/frames/<batch>_frames.json``.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from app.evaluation import EVAL_DIR
from app.evaluation.experiment import TopicData, load_topic
from app.providers.llm import BudgetedLLM
from app.reasoning import frames as F

FRAMES_DIR = EVAL_DIR / "frames"


def frames_path(batch: str) -> Path:
    return FRAMES_DIR / f"{batch}_frames.json"


def batch_claims(batch: str) -> tuple[dict[tuple[str, str], TopicData], list[tuple[str, str, str]]]:
    """Topic data and the distinct (run, topic, claim_id) of every claim in a pair of the annotation batch."""
    ref = json.loads((EVAL_DIR / "annotation" / batch / "reference.json").read_text())["items"]
    datas: dict[tuple[str, str], TopicData] = {}
    seen: dict[tuple[str, str, str], None] = {}
    for r in ref.values():
        key = (r["run"], r["topic"])
        if key not in datas:
            datas[key] = load_topic(*key)
        rec = next(x for x in datas[key].records if x["pair_id"] == r["pair_id"])
        seen[(*key, rec["a"])] = None
        seen[(*key, rec["b"])] = None
    return datas, list(seen)


def load_raw(batch: str) -> dict[str, dict]:
    path = frames_path(batch)
    return json.loads(path.read_text())["frames"] if path.exists() else {}


def extract(batch: str, llm: BudgetedLLM, *, limit: int | None = None, concurrency: int = 6) -> dict[str, int]:
    datas, claims = batch_claims(batch)
    raw = load_raw(batch)
    todo = [c for c in claims if "/".join(c) not in raw][:limit]

    def call(item: tuple[str, str, str]) -> tuple[str, dict | None]:
        run, topic, cid = item
        data = datas[(run, topic)]
        claim = data.claims[cid]
        return "/".join(item), llm.complete_json(F.SYSTEM, F.render_claim(claim, data.papers.get(claim.paper_id)), F.SCHEMA, max_tokens=F.MAX_TOKENS)

    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for key, out in pool.map(call, todo):
            if out:
                raw[key] = out
            else:
                failed += 1
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    frames_path(batch).write_text(json.dumps({
        "batch": batch, "model": getattr(llm.provider, "model", None), "protocol": "one call per claim; the model sees one claim, its passage and its abstract",
        "frames": raw}, indent=1, ensure_ascii=False))
    return {"claims": len(claims), "requested": len(todo), "failed": failed, "stored": len(raw)}


def load_frames(batch: str) -> dict[tuple[str, str, str], F.ClaimFrame]:
    """Validated, quote-checked frames by (run, topic, claim_id)."""
    datas, claims = batch_claims(batch)
    raw = load_raw(batch)
    out = {}
    for item in claims:
        data = datas[item[:2]]
        claim = data.claims[item[2]]
        frame = F.frame_from_output(raw.get("/".join(item)), claim, data.papers.get(claim.paper_id))
        if frame:
            out[item] = frame
    return out


def key_usage(api_key: str) -> float | None:
    """US$ spent so far on the OpenRouter key (None if it cannot be read). The key itself is never printed."""
    try:
        r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        return float(r.json()["data"]["usage"])
    except Exception:  # noqa: BLE001
        return None
