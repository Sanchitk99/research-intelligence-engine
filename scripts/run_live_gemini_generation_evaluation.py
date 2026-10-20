"""Run the 28-question generation benchmark against Gemini without overwriting baselines.

This runner deliberately supplies the provider model to GenerationConfig.  The
generic benchmark runner's default GenerationConfig is OpenAI-oriented, so a
live Gemini run must not rely on that default when verifying model selection.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.evaluation.generation_evaluator import (
    GenerationEvaluationError,
    GenerationEvaluationSummary,
    GenerationEvaluator,
    save_generation_results,
)
from src.generation.generator import AnswerGenerator
from src.generation.models import GenerationConfig
from src.generation.providers.factory import get_llm_provider


RESULTS_DIR = ROOT / "evaluation" / "results"
ANSWERABLE_PATH = ROOT / "evaluation" / "questions" / "answerable_questions.json"
OOS_PATH = ROOT / "evaluation" / "questions" / "out_of_scope_questions.json"
DEFAULT_MODEL = "gemini-flash-lite-latest"


def redact(value: Any) -> str:
    """Return an error string with the configured API key removed."""
    text = str(value)
    secret = os.environ.get("GEMINI_API_KEY")
    if secret:
        text = text.replace(secret, "[REDACTED_API_KEY]")
    return text


def configured_model(model_arg: Optional[str]) -> str:
    return model_arg or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def validate_dataset() -> int:
    with ANSWERABLE_PATH.open("r", encoding="utf-8") as handle:
        answerable = json.load(handle)
    with OOS_PATH.open("r", encoding="utf-8") as handle:
        out_of_scope = json.load(handle)
    if len(answerable) != 18 or len(out_of_scope) != 10:
        raise RuntimeError(
            f"Unexpected evaluation set size: answerable={len(answerable)}, out_of_scope={len(out_of_scope)}; expected 18 + 10."
        )
    return len(answerable) + len(out_of_scope)


def write_failure(mode: str, model: str, planned_calls: int, error: BaseException) -> Path:
    question_id = getattr(error, "question_id", "unknown")
    category = getattr(error, "category", "unknown")
    stage = getattr(error, "stage", "unknown")
    failure_type = type(error).__name__
    question_order = []
    with ANSWERABLE_PATH.open("r", encoding="utf-8") as handle:
        question_order.extend(item["id"] for item in json.load(handle))
    with OOS_PATH.open("r", encoding="utf-8") as handle:
        question_order.extend(item["id"] for item in json.load(handle))
    if question_id in question_order:
        question_position = question_order.index(question_id) + 1
        questions_attempted = question_position
        questions_completed = max(0, question_position - 1)
        questions_failed = 1
        provider_call_opportunities_attempted = question_position + 1  # warmup + benchmark calls through the failure
        provider_calls_completed = question_position
        provider_calls_failed = 1
    else:
        questions_attempted = 0
        questions_completed = 0
        questions_failed = 0
        provider_call_opportunities_attempted = 1
        provider_calls_completed = 0
        provider_calls_failed = 1
    payload = {
        "status": "failed",
        "timestamp": datetime.now().isoformat(),
        "provider": "gemini",
        "model": model,
        "retrieval_mode": mode,
        "planned_provider_call_opportunities": planned_calls,
        "questions_attempted": questions_attempted,
        "questions_completed": questions_completed,
        "questions_failed": questions_failed,
        "provider_call_opportunities_attempted": provider_call_opportunities_attempted,
        "provider_calls_completed": provider_calls_completed,
        "provider_calls_failed": provider_calls_failed,
        "question_id": question_id,
        "category": category,
        "failure_stage": stage,
        "failure_type": failure_type,
        "failure_message": redact(error),
    }
    json_path = RESULTS_DIR / f"generation_evaluation_gemini_{mode}_failure.json"
    md_path = RESULTS_DIR / f"generation_evaluation_gemini_{mode}_failure.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(
        "# Live Gemini Generation Evaluation — Failed\n\n"
        f"- **Mode:** `{mode}`\n"
        f"- **Provider:** `gemini`\n"
        f"- **Model:** `{model}`\n"
        f"- **Failure type:** `{failure_type}`\n"
        f"- **Failure stage:** `{stage}`\n"
        f"- **Question:** `{question_id}` ({category})\n"
        f"- **Questions:** {questions_attempted} attempted, {questions_completed} completed, {questions_failed} failed\n"
        f"- **Provider calls:** {provider_call_opportunities_attempted} opportunities attempted, {provider_calls_completed} completed, {provider_calls_failed} failed\n"
        f"- **Failure:** {redact(error)}\n\n"
        "The live evaluation stopped without treating the partial run as a successful benchmark.\n",
        encoding="utf-8",
    )
    return json_path


def run_mode(mode: str, model: str, top_k: int) -> None:
    expected_questions = validate_dataset()
    planned_calls = expected_questions + 1  # one warmup plus the 28 benchmark questions
    requested_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if requested_provider != "gemini":
        raise RuntimeError(f"LLM_PROVIDER must be exactly 'gemini' for this run; got '{requested_provider or 'unset'}'.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured; refusing to fall back to mock.")

    provider = get_llm_provider(provider_type="gemini", model_name=model)
    if provider.name != "gemini":
        raise RuntimeError(f"Provider verification failed: active provider is '{provider.name}', not 'gemini'.")
    if provider.default_model != model:
        raise RuntimeError(
            f"Model verification failed: configured '{model}' but provider reports '{provider.default_model}'."
        )

    print(f"Active provider verified: {provider.name}; model verified: {provider.default_model}; mode: {mode}")
    print(f"Provider-call opportunities for this mode: {planned_calls} (1 warmup + {expected_questions} questions); retries: 0")

    config = GenerationConfig(
        provider_type="gemini",
        model_name=provider.default_model,
        top_k=top_k,
    )
    generator = AnswerGenerator(provider=provider, config=config, retrieval_mode=mode)
    try:
        evaluator = GenerationEvaluator(generator=generator)
        summary, results = evaluator.evaluate_all(
            answerable_path=ANSWERABLE_PATH,
            out_of_scope_path=OOS_PATH,
            top_k=top_k,
            warmup=True,
            cold_start_measurement_mode="isolated_subprocess",
            fail_fast_on_generation_error=True,
        )
    except Exception as error:
        path = write_failure(mode, model, planned_calls, error)
        print(f"LIVE EVALUATION FAILED; no success artifact was written. Failure artifact: {path}")
        print(f"Failure: {redact(error)}")
        raise
    finally:
        generator.close()

    json_path = RESULTS_DIR / f"generation_evaluation_gemini_{mode}.json"
    md_path = RESULTS_DIR / f"generation_evaluation_gemini_{mode}_report.md"
    save_generation_results(summary, results, json_path, md_path)
    print(f"Completed {summary.total_evaluated} questions; successful calls: {summary.successful_generation_calls}; failed calls: {summary.failed_generation_calls}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


def load_summary(mode: str) -> GenerationEvaluationSummary:
    path = RESULTS_DIR / f"generation_evaluation_gemini_{mode}.json"
    with path.open("r", encoding="utf-8") as handle:
        return GenerationEvaluationSummary.model_validate(json.load(handle)["summary"])


def compare_mode(mode: str, live: GenerationEvaluationSummary) -> Dict[str, Any]:
    baseline_path = RESULTS_DIR / f"generation_evaluation_{mode}.json"
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)["summary"]
    comparable = [
        "oos_abstention_accuracy",
        "answerable_completion_rate",
        "target_evidence_supported_rate",
        "evidence_claim_supported_rate",
        "expected_answer_lexical_match_rate",
        "appropriate_outcome_rate",
        "unsupported_answer_risk_count",
        "citation_validity_rate",
        "citation_excerpt_faithfulness_rate",
    ]
    delta = {}
    for key in comparable:
        delta[key] = round(getattr(live, key) - baseline[key], 4)
    return {
        "mode": mode,
        "live_gemini": live.model_dump(),
        "mock_baseline": baseline,
        "directly_comparable_metrics": comparable,
        "delta_live_minus_mock": delta,
        "latency_comparison_note": "Latency is recorded for both runs but is not a provider-quality metric; Gemini includes network/API time while mock latency is simulated/offline.",
        "token_comparison_note": "Token usage is provider-reported for Gemini; the historical mock baseline has no comparable token accounting.",
    }


def write_comparison(dense: GenerationEvaluationSummary, hybrid: GenerationEvaluationSummary) -> None:
    modes = [compare_mode("dense", dense), compare_mode("hybrid", hybrid)]
    payload = {
        "timestamp": datetime.now().isoformat(),
        "provider": "gemini",
        "model": dense.model_name,
        "question_count": 28,
        "planned_provider_call_opportunities": 58,
        "warmup_calls": 2,
        "benchmark_call_opportunities": 56,
        "modes": modes,
    }
    json_path = RESULTS_DIR / "generation_comparison_gemini_vs_mock.json"
    md_path = RESULTS_DIR / "generation_comparison_gemini_vs_mock_report.md"
    lines = [
        "# Live Gemini vs Mock Generation Evaluation",
        "",
        f"**Provider:** `gemini`  **Model:** `{dense.model_name}`  **Question set:** 28 (18 answerable, 10 out-of-scope)",
        "",
        "This report preserves the historical mock artifacts and compares metrics produced by the same evaluation code, dataset, retrieval modes, and scoring rules. Quality metrics below are directly comparable; latency is operationally reported but provider-dependent, and token usage is not comparable because the mock baseline did not record provider tokens.",
        "",
        "**Call plan:** 58 provider-call opportunities total (2 warmups + 56 benchmark questions), zero retries. Retrieval-empty questions, if any, skip the provider call and are recorded in each live summary.",
        "",
    ]
    for entry in modes:
        live = entry["live_gemini"]
        mock = entry["mock_baseline"]
        lines.extend([
            f"## {entry['mode'].capitalize()} retrieval",
            "",
            "| Metric | Live Gemini | Mock baseline | Delta (live - mock) |",
            "|---|---:|---:|---:|",
        ])
        for key in entry["directly_comparable_metrics"]:
            live_value = live[key]
            mock_value = mock[key]
            if key == "unsupported_answer_risk_count":
                live_text, mock_text = str(live_value), str(mock_value)
                delta_text = f"{entry['delta_live_minus_mock'][key]:+.0f}"
            else:
                live_text, mock_text = f"{live_value * 100:.1f}%", f"{mock_value * 100:.1f}%"
                delta_text = f"{entry['delta_live_minus_mock'][key] * 100:+.1f} pp"
            lines.append(f"| `{key}` | {live_text} | {mock_text} | {delta_text} |")
        lines.extend([
            "",
            f"- Live calls: {live['successful_generation_calls']} successful, {live['failed_generation_calls']} failed, {live['retrieval_empty_count']} retrieval-empty skips.",
            f"- Live latency: cold start {live['cold_start_latency_ms']} ms; warm mean {live['warm_mean_latency_ms']} ms; warm median {live['warm_median_latency_ms']} ms.",
            f"- Live tokens: prompt {live['prompt_tokens_total']}; completion {live['completion_tokens_total']}; total {live['total_tokens']} across {live['token_usage_available_calls']} calls with usage metadata.",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Gemini 28-question generation evaluation")
    parser.add_argument("--mode", choices=["dense", "hybrid", "both"], default="both")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    model = configured_model(args.model)

    if args.mode == "both" and not args.worker:
        print("Planned live provider calls: 58 maximum (2 warmups + 56 benchmark questions); retries: 0")
        for mode in ("dense", "hybrid"):
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--model", model, "--top-k", str(args.top_k), "--worker"]
            completed = subprocess.run(command, cwd=ROOT)
            if completed.returncode != 0:
                return completed.returncode
        write_comparison(load_summary("dense"), load_summary("hybrid"))
        return 0

    try:
        validate_dataset()
        run_mode(args.mode, model, args.top_k)
    except Exception as error:
        print(f"Stopped: {redact(error)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
