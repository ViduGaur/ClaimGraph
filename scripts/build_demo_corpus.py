#!/usr/bin/env python3
"""Build the demo corpus from real arXiv metadata (title, authors, year, DOI, abstract), which arXiv releases under CC0.

Topic: "Do Vision Transformers need large-scale pre-training to beat CNNs?", chosen because the literature has
well-known apparent contradictions that depend on context (data scale, augmentation, training recipe).

Usage (one request, within arXiv's 1 request / 3 s guidance):
    backend/.venv/bin/python scripts/build_demo_corpus.py
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx  # arXiv rejects urllib's default request shape (HTTP 406)

ARXIV_IDS = [
    "2010.11929",  # ViT: An Image is Worth 16x16 Words
    "2012.12877",  # DeiT: Training data-efficient image transformers
    "2106.10270",  # How to train your ViT? (AugReg)
    "2106.01548",  # When Vision Transformers Outperform ResNets without Pretraining
    "2112.13492",  # Vision Transformer for Small-Size Datasets
    "2104.05704",  # Escaping the Big Data Paradigm with Compact Transformers
    "2106.03746",  # Efficient Training of Visual Transformers with Small Datasets
    "2205.01580",  # Better plain ViT baselines for ImageNet-1k
    "2201.03545",  # ConvNeXt: A ConvNet for the 2020s
    "2310.16764",  # ConvNets Match Vision Transformers at Scale
]

QUESTION = "Do vision transformers need large-scale pre-training to outperform convolutional networks?"
OUT = Path(__file__).resolve().parents[1] / "backend" / "app" / "demo" / "corpus.json"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
UA = "ClaimGraph-demo-builder/0.1 (student research project)"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch(ids: list[str]) -> list[dict]:
    resp = httpx.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        headers={"User-Agent": UA},
        follow_redirects=True,
        timeout=60,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    papers = []
    for entry in root.findall("a:entry", NS):
        abs_url = entry.findtext("a:id", default="", namespaces=NS)
        arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
        journal = entry.findtext("arxiv:journal_ref", default="", namespaces=NS)
        category = entry.find("arxiv:primary_category", NS)
        papers.append(
            {
                "source_ids": {"arxiv": arxiv_id},
                "title": _clean(entry.findtext("a:title", default="", namespaces=NS)),
                "authors": [_clean(a.findtext("a:name", default="", namespaces=NS)) for a in entry.findall("a:author", NS)],
                "year": int(entry.findtext("a:published", default="0000", namespaces=NS)[:4]),
                "venue": _clean(journal) or f"arXiv ({category.get('term') if category is not None else 'preprint'})",
                "doi": f"10.48550/arXiv.{arxiv_id}",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "abstract": _clean(entry.findtext("a:summary", default="", namespaces=NS)),
                "is_open_access": True,
                "oa_status": "green",
                "license_note": "arXiv metadata (CC0)",
            }
        )
    return papers


def main() -> None:
    papers = fetch(ARXIV_IDS)
    time.sleep(3)
    order = {pid: i for i, pid in enumerate(ARXIV_IDS)}
    papers.sort(key=lambda p: order.get(p["source_ids"]["arxiv"], 99))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"question": QUESTION, "papers": papers}, indent=2, ensure_ascii=False))
    print(f"wrote {len(papers)} papers -> {OUT}")


if __name__ == "__main__":
    main()
