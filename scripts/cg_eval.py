#!/usr/bin/env python3
"""ClaimGraph evaluation CLI — see eval/README.md.

    backend/.venv/bin/python scripts/cg_eval.py <command> [options]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
