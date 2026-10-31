#!/usr/bin/env python3
"""Run the ClaimGraph pipeline from the terminal and print everything the claim graph is built from:
claims, their contexts, the related claim pairs with similarity / NLI / context factors, the relation,
the conflict verdict and its evidence.

    backend/.venv/bin/python scripts/cg_run.py "Does X improve Y?" [--max-papers 8] [--no-llm] [--json out.json]
    backend/.venv/bin/python scripts/cg_run.py --demo            # bundled demo corpus, no network search

The LLM (OpenRouter, CLAIMGRAPH_LLM_API_KEY in .env) is used when configured; --no-llm forces the
rule-based pipeline.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import Settings  # noqa: E402
from app.evaluation.report import format_analysis  # noqa: E402
from app.pipelines.analysis import AnalysisPipeline, PipelineError  # noqa: E402
from app.schemas.api import AnalyzeRequest  # noqa: E402

DEMO_QUESTION = "Do vision transformers need large-scale pre-training to outperform convolutional networks?"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("question", nargs="?", default=None)
parser.add_argument("--demo", action="store_true", help="use the bundled demo corpus")
parser.add_argument("--max-papers", type=int, default=None)
parser.add_argument("--no-llm", action="store_true")
parser.add_argument("--llm-calls", type=int, default=None, help="LLM call budget for this run")
parser.add_argument("--json", help="also write the full result document to this file")
args = parser.parse_args()
question = args.question or (DEMO_QUESTION if args.demo else None)
if not question:
    parser.error("give a research question, or --demo")

overrides = {"demo_mode": args.demo}
if args.llm_calls:
    overrides["llm_max_calls"] = args.llm_calls
settings = Settings(**overrides)
warnings: list[str] = []
pipeline = AnalysisPipeline(settings, progress=lambda k, st, d: print(f"  [{k}] {st} {d or ''}", file=sys.stderr),
                            warn=lambda stage, msg: warnings.append(f"[{stage}] {msg}"))
try:
    result = pipeline.run(AnalyzeRequest(question=question, max_papers=args.max_papers, use_llm=False if args.no_llm else None))
except PipelineError as exc:
    sys.exit(f"analysis failed: {exc}")
print(format_analysis(result))
for w in warnings:
    print("warning:", w)
if args.json:
    Path(args.json).write_text(result.model_dump_json(indent=1))
    print(f"\nfull result written to {args.json}")
