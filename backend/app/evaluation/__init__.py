"""Evaluation harness: frozen topic corpora → claims → candidate pairs with all signals →
baseline / ablation / full-system predictions → blind annotation → metrics, tables and studies.

Everything here reuses the production pipeline modules; nothing in ``app.evaluation`` changes
how an analysis runs. Entry point: ``scripts/cg_eval.py`` (see ``eval/README.md``).

Modules by purpose:

* harness: ``corpus``, ``experiment``, ``llm_baseline``, ``scoring``, ``metrics``, ``plots``, ``report``, ``cli``
* annotation protocols: ``annotation`` (two questions, human sheets), ``model_annotation`` (five relations, model
  annotators), ``conflict_annotation`` (four states, model annotators)
* checks on the system's output: ``explanation_audit``, ``rejections``
* studies: ``comparability_study``, ``fourstate_study``, ``frame_extraction``, ``layer_study``, ``fresh_validation``
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import DATA_DIR, REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval"  # tracked: topics, claims, pairs, predictions, labels, results
CORPORA_DIR = DATA_DIR / "eval" / "corpora"  # not tracked: full paper text snapshots
LLM_CACHE_PATH = EVAL_DIR / "llm_cache.jsonl"
TOPICS_PATH = EVAL_DIR / "topics.json"


def run_dir(run: str) -> Path:
    return EVAL_DIR / "runs" / run
