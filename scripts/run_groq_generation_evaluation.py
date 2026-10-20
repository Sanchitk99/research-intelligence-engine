"""Dry-run or run the resumable Groq generation evaluation."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.evaluation.groq_generation_runner import DEFAULT_MODEL, preflight, run_mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable Groq 28-question generation evaluation")
    parser.add_argument("--mode", choices=["dense", "hybrid", "both"], default="both")
    parser.add_argument("--model", default=os.environ.get("GROQ_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-calls", type=int, default=28, help="Maximum provider calls per retrieval mode")
    parser.add_argument("--preflight", "--dry-run", action="store_true", help="Calculate calls and write preflight artifacts without generation")
    parser.add_argument("--resume", action="store_true", help="Resume existing compatible mode checkpoints")
    args = parser.parse_args()

    modes = ["dense", "hybrid"] if args.mode == "both" else [args.mode]
    if args.max_calls < 0:
        parser.error("--max-calls must be non-negative")
    for mode in modes:
        report = preflight(mode, args.model, args.top_k, args.max_calls)
        print(f"{mode}: {report['question_count']} questions; {report['expected_provider_calls']} expected provider calls; warmups=0; budget={args.max_calls}")
    if args.preflight:
        print("Preflight only: generation calls made=0")
        return 0
    for mode in modes:
        result = run_mode(mode, args.model, args.top_k, args.max_calls, resume=args.resume)
        progress = result["progress"]
        print(f"{mode}: status={result['status']}; completed={progress['completed_questions']}; failed={progress['failed_questions']}; remaining={progress['remaining_questions']}; calls={progress['provider_calls_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
