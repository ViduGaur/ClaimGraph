#!/usr/bin/env python3
"""Check the files named in eval/FREEZE.json against the hashes recorded there. Read-only.

    python eval/verify_freeze.py

Checks, in this order:

1. every path in the ``sha256`` blocks of FREEZE.json (the initial freeze and the addenda), against the LAST hash
   recorded for that path;
2. the file hashes written into the text of the addenda (the pre-registration, the fresh-topic result files and the
   scoring code);
3. the ``scorer_sha256`` stored in the fresh-topic key and result files, against the scoring code.

Prints OK, DIFF or MISSING for each file and exits 1 if any is not OK. It never writes anything and never updates a
hash; a DIFF means the file changed after the hash was recorded, and FREEZE.json is the record of when that was known.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "eval" / "FREEZE.json"
SCORER = "backend/app/evaluation/fresh_validation.py"
SCORER_COPIES = ("eval/annotation/f1/f1_key.json", "eval/results/f1_validation/validation.json")
# "<file> sha256 <hex>", "<file> (sha256 <hex>)" or, within one sentence, "<file> <hex>"
TEXT_HASH = re.compile(r"([\w./-]+\.(?:py|md|json|jsonl))\)?,? (?:\(?sha256 )?([0-9a-f]{64})")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(status: str, name: str, detail: str = "") -> bool:
    print(f"{status:<8}{name}" + (f"   {detail}" if detail else ""))
    return status == "OK"


def recorded_blocks(freeze: dict):
    """(label, {path: hash}) for the initial freeze and every addendum that carries hashes."""
    yield f"frozen_at {freeze['frozen_at']}", freeze["sha256"]
    for a in freeze.get("addenda", []):
        if a.get("sha256"):
            yield f"addendum {a['at']}", a["sha256"]


def check_blocks(freeze: dict) -> list[bool]:
    latest: dict[str, tuple[str, str]] = {}
    for label, block in recorded_blocks(freeze):
        for path, digest in block.items():
            latest[path] = (digest, label)
    print(f"== {len(latest)} paths with a recorded hash (last recorded hash of each)")
    results = []
    for path, (digest, label) in latest.items():
        f = ROOT / path
        if not f.is_file():
            results.append(report("MISSING", path, f"recorded in {label}"))
        elif sha256(f) == digest:
            results.append(report("OK", path, f"recorded in {label}"))
        else:
            results.append(report("DIFF", path, f"last recorded in {label}: {digest[:12]}, file is {sha256(f)[:12]}"))
    return results


def check_text_hashes(freeze: dict) -> list[bool]:
    print("== hashes written into the text of the addenda")
    results, seen = [], set()
    for a in freeze.get("addenda", []):
        for name, digest in TEXT_HASH.findall(a["what"]):
            if (name, digest) in seen:
                continue
            seen.add((name, digest))
            candidates = [ROOT / name] if "/" in name else sorted((ROOT / "eval").rglob(name)) + sorted((ROOT / "backend").rglob(name))
            candidates = [c for c in candidates if c.is_file()]
            where = f"addendum {a['at']}"
            if not candidates:
                results.append(report("MISSING", name, where))
                continue
            match = next((c for c in candidates if sha256(c) == digest), None)
            if match:
                results.append(report("OK", str(match.relative_to(ROOT)), where))
            else:
                results.append(report("DIFF", str(candidates[0].relative_to(ROOT)), f"{where}: recorded {digest[:12]}"))
    return results


def find_key(obj, key: str):
    """First value stored under ``key`` at any depth, or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for item in obj:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


def check_scorer_copies() -> list[bool]:
    print("== scorer_sha256 stored with the fresh-topic results")
    scorer = ROOT / SCORER
    if not scorer.is_file():
        return [report("MISSING", SCORER)]
    results = []
    for rel in SCORER_COPIES:
        f = ROOT / rel
        if not f.is_file():
            results.append(report("MISSING", rel))
            continue
        stored = find_key(json.loads(f.read_text()), "scorer_sha256")
        results.append(report("OK" if stored == sha256(scorer) else "DIFF", rel, "scorer_sha256 vs " + SCORER))
    return results


def main() -> int:
    freeze = json.loads(FREEZE.read_text())
    results = check_blocks(freeze) + check_text_hashes(freeze) + check_scorer_copies()
    bad = results.count(False)
    print(f"\n{len(results) - bad} OK, {bad} not OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
