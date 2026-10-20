"""CLI runner for Answer Generation evaluation benchmark (Dense, Hybrid, and Comparative)."""

import sys
import json
import argparse
import subprocess
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.generation.generator import AnswerGenerator
from src.generation.providers.factory import get_llm_provider
from src.evaluation.generation_evaluator import (
    GenerationEvaluator,
    GenerationEvaluationSummary,
    save_generation_results,
    generate_comparison_report,
)


def run_mode_eval(mode: str, provider, execution_isolation: str = "isolated_subprocess"):
    """Run generation evaluation for a specific retrieval mode ('dense' or 'hybrid')."""
    print(f"\n---> Initializing AnswerGenerator with '{mode}' retrieval mode...")
    generator = AnswerGenerator(provider=provider, retrieval_mode=mode)
    evaluator = GenerationEvaluator(generator=generator)

    print(f"---> Running evaluation across answerable and out-of-scope questions ({mode})...")
    summary, results = evaluator.evaluate_all(cold_start_measurement_mode=execution_isolation)
    generator.close()
    return summary, results


def print_summary_table(summary):
    print("\n" + "=" * 76)
    print(f" AGGREGATE GENERATION METRICS ({summary.retrieval_mode.upper()} RETRIEVAL)")
    print("=" * 76)
    print(f"Total Evaluated Questions:             {summary.total_evaluated}")
    print(f"Out-of-Scope Abstention Accuracy:      {summary.oos_abstention_accuracy * 100:.1f}% ({int(round(summary.oos_abstention_accuracy * summary.total_out_of_scope))} / {summary.total_out_of_scope})")
    print(f"Answerable Completion Rate:            {summary.answerable_completion_rate * 100:.1f}% ({int(round(summary.answerable_completion_rate * summary.total_answerable))} / {summary.total_answerable})")
    print(f"Target Provenance Citation Rate:       {summary.target_evidence_supported_rate * 100:.1f}% ({int(round(summary.target_evidence_supported_rate * summary.total_answerable))} / {summary.total_answerable})")
    print(f"Evidence Claim Supported Rate:         {summary.evidence_claim_supported_rate * 100:.1f}% ({int(round(summary.evidence_claim_supported_rate * summary.total_answerable))} / {summary.total_answerable})")
    print(f"Expected Answer Match Rate (Lexical):  {summary.expected_answer_lexical_match_rate * 100:.1f}% ({int(round(summary.expected_answer_lexical_match_rate * summary.total_answerable))} / {summary.total_answerable})")
    print(f"Appropriate Outcome Rate (Strict):     {summary.appropriate_outcome_rate * 100:.1f}% ({int(round(summary.appropriate_outcome_rate * summary.total_evaluated))} / {summary.total_evaluated})")
    print(f"Unsupported Answer Risk Count:         {summary.unsupported_answer_risk_count} / {summary.total_answerable}")
    print(f"Citation Validity Rate:                {summary.citation_validity_rate * 100:.1f}% ({summary.total_valid_citations} / {summary.total_citations_emitted})")
    print(f"Citation Excerpt Faithfulness:         {summary.citation_excerpt_faithfulness_rate * 100:.1f}% ({summary.total_faithful_excerpts} / {summary.total_citations_emitted})")
    if summary.cold_start_latency_ms is not None:
        mode_note = f" ({summary.cold_start_measurement_mode})" if getattr(summary, "cold_start_measurement_mode", None) else ""
        print(f"Cold-Start Latency:                    {summary.cold_start_latency_ms:.1f}ms{mode_note}")
    w_mean = summary.warm_mean_latency_ms if summary.warm_mean_latency_ms is not None else summary.mean_latency_ms
    print(f"Warm Mean Generation Latency:          {w_mean:.1f}ms")
    if summary.warm_median_latency_ms is not None:
        print(f"Warm Median Generation Latency:        {summary.warm_median_latency_ms:.1f}ms")
    print("=" * 76)


def print_comparison_table(dense_summary, hybrid_summary, comp_report_path, comp_json_path):
    print("\n" + "=" * 76)
    print(" SIDE-BY-SIDE RETRIEVAL MODE COMPARISON (GENERATION BENCHMARK)")
    print("=" * 76)
    print(f"{'Metric':<38} | {'Dense':<14} | {'Hybrid':<14} | {'Delta':<8}")
    print("-" * 76)
    print(f"{'Target Provenance Citation Rate':<38} | {dense_summary.target_evidence_supported_rate*100:>5.1f}% ({int(round(dense_summary.target_evidence_supported_rate*18))}/18) | {hybrid_summary.target_evidence_supported_rate*100:>5.1f}% ({int(round(hybrid_summary.target_evidence_supported_rate*18))}/18) | {(hybrid_summary.target_evidence_supported_rate - dense_summary.target_evidence_supported_rate)*100:>+5.1f}%")
    print(f"{'Evidence Claim Supported Rate':<38} | {dense_summary.evidence_claim_supported_rate*100:>5.1f}% ({int(round(dense_summary.evidence_claim_supported_rate*18))}/18) | {hybrid_summary.evidence_claim_supported_rate*100:>5.1f}% ({int(round(hybrid_summary.evidence_claim_supported_rate*18))}/18) | {(hybrid_summary.evidence_claim_supported_rate - dense_summary.evidence_claim_supported_rate)*100:>+5.1f}%")
    print(f"{'Expected Answer Match Rate':<38} | {dense_summary.expected_answer_lexical_match_rate*100:>5.1f}% ({int(round(dense_summary.expected_answer_lexical_match_rate*18))}/18) | {hybrid_summary.expected_answer_lexical_match_rate*100:>5.1f}% ({int(round(hybrid_summary.expected_answer_lexical_match_rate*18))}/18) | {(hybrid_summary.expected_answer_lexical_match_rate - dense_summary.expected_answer_lexical_match_rate)*100:>+5.1f}%")
    print(f"{'Strict Appropriate Outcome Rate':<38} | {dense_summary.appropriate_outcome_rate*100:>5.1f}% ({int(round(dense_summary.appropriate_outcome_rate*28))}/28) | {hybrid_summary.appropriate_outcome_rate*100:>5.1f}% ({int(round(hybrid_summary.appropriate_outcome_rate*28))}/28) | {(hybrid_summary.appropriate_outcome_rate - dense_summary.appropriate_outcome_rate)*100:>+5.1f}%")
    print(f"{'Unsupported Answer Risk Count':<38} | {dense_summary.unsupported_answer_risk_count:>2d}/18           | {hybrid_summary.unsupported_answer_risk_count:>2d}/18           | {hybrid_summary.unsupported_answer_risk_count - dense_summary.unsupported_answer_risk_count:>+2d}")
    d_cold = f"{dense_summary.cold_start_latency_ms:>6.1f}ms" if dense_summary.cold_start_latency_ms is not None else "   N/A  "
    h_cold = f"{hybrid_summary.cold_start_latency_ms:>6.1f}ms" if hybrid_summary.cold_start_latency_ms is not None else "   N/A  "
    d_mode = getattr(dense_summary, "cold_start_measurement_mode", None) or "isolated_subprocess"
    h_mode = getattr(hybrid_summary, "cold_start_measurement_mode", None) or "isolated_subprocess"
    if d_mode != "in_process" and h_mode != "in_process":
        delta_cold = f"{(hybrid_summary.cold_start_latency_ms - dense_summary.cold_start_latency_ms):>+6.1f}ms" if (dense_summary.cold_start_latency_ms is not None and hybrid_summary.cold_start_latency_ms is not None) else "  N/A   "
    else:
        delta_cold = " Non-comp*"
    print(f"{'Cold-Start Latency':<38} | {d_cold}       | {h_cold}       | {delta_cold}")

    d_wm = dense_summary.warm_mean_latency_ms if dense_summary.warm_mean_latency_ms is not None else dense_summary.mean_latency_ms
    h_wm = hybrid_summary.warm_mean_latency_ms if hybrid_summary.warm_mean_latency_ms is not None else hybrid_summary.mean_latency_ms
    print(f"{'Warm Mean Generation Latency':<38} | {d_wm:>6.1f}ms       | {h_wm:>6.1f}ms       | {h_wm - d_wm:>+6.1f}ms")

    d_wmed = f"{dense_summary.warm_median_latency_ms:>6.1f}ms" if dense_summary.warm_median_latency_ms is not None else "   N/A  "
    h_wmed = f"{hybrid_summary.warm_median_latency_ms:>6.1f}ms" if hybrid_summary.warm_median_latency_ms is not None else "   N/A  "
    delta_wmed = f"{(hybrid_summary.warm_median_latency_ms - dense_summary.warm_median_latency_ms):>+6.1f}ms" if (dense_summary.warm_median_latency_ms is not None and hybrid_summary.warm_median_latency_ms is not None) else "  N/A   "
    print(f"{'Warm Median Generation Latency':<38} | {d_wmed}       | {h_wmed}       | {delta_wmed}")
    print("=" * 76)
    print(f"Comparison Markdown Report: {comp_report_path}")
    print(f"Comparison JSON Artifact:    {comp_json_path}")
    print("=" * 76)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Answer Generation Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "both"],
        default="both",
        help="Retrieval mode to evaluate: 'dense', 'hybrid', or 'both' (default: both)",
    )
    parser.add_argument(
        "--isolated-subprocess",
        dest="isolated_subprocess",
        action="store_true",
        default=True,
        help="Execute dense and hybrid evaluations in separate subprocesses for fair, independent cold-start latency measurement (default: True)",
    )
    parser.add_argument(
        "--no-isolated-subprocess",
        dest="isolated_subprocess",
        action="store_false",
        help="Execute evaluations sequentially in the same process (warning: biases second mode cold-start)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal worker flag
    )
    args = parser.parse_args()

    # If --mode both is requested with isolated subprocesses enabled, orchestrate via clean child processes
    if args.mode == "both" and args.isolated_subprocess and not args.worker:
        print("=" * 76)
        print(" Research Intelligence Engine -- Generation Evaluation Benchmark (Task 4.2.3)")
        print("=" * 76)
        print("Execution Mode: Both (Dense & Hybrid)")
        print("Isolation: Independent Python Subprocesses (Ensuring Unbiased Cold-Start Timing)")
        print("-" * 76)

        script_path = str(Path(__file__).resolve())

        print("\n[Phase 1/2] Launching independent subprocess for 'dense' retrieval...")
        subprocess.run([sys.executable, script_path, "--mode", "dense", "--worker"], check=True)

        print("\n[Phase 2/2] Launching independent subprocess for 'hybrid' retrieval...")
        subprocess.run([sys.executable, script_path, "--mode", "hybrid", "--worker"], check=True)

        dense_json_path = Path("evaluation/results/generation_evaluation_dense.json")
        hybrid_json_path = Path("evaluation/results/generation_evaluation_hybrid.json")

        with open(dense_json_path, "r", encoding="utf-8") as f:
            dense_summary = GenerationEvaluationSummary.model_validate(json.load(f)["summary"])
        with open(hybrid_json_path, "r", encoding="utf-8") as f:
            hybrid_summary = GenerationEvaluationSummary.model_validate(json.load(f)["summary"])

        comp_report_path = Path("evaluation/results/generation_comparison_report.md")
        comp_json_path = Path("evaluation/results/generation_comparison.json")
        generate_comparison_report(dense_summary, hybrid_summary, comp_report_path, comp_json_path)
        print_comparison_table(dense_summary, hybrid_summary, comp_report_path, comp_json_path)
        return

    # Direct single-mode or in-process execution
    print("=" * 76)
    print(" Research Intelligence Engine -- Generation Evaluation Benchmark (Task 4.2.3)")
    print("=" * 76)

    provider = get_llm_provider()
    print(f"Using Provider: {provider.name} (Model: {provider.default_model})")
    print(f"Evaluation Mode: {args.mode}")

    isolation_mode = "isolated_subprocess" if (args.worker or args.mode != "both") else "in_process"
    dense_summary = None
    hybrid_summary = None

    if args.mode in ("dense", "both"):
        dense_summary, dense_results = run_mode_eval("dense", provider, execution_isolation=isolation_mode)
        save_generation_results(
            dense_summary,
            dense_results,
            Path("evaluation/results/generation_evaluation_dense.json"),
            Path("evaluation/results/generation_evaluation_dense_report.md"),
        )
        # Also save to default paths for backwards compatibility
        save_generation_results(
            dense_summary,
            dense_results,
            Path("evaluation/results/generation_evaluation.json"),
            Path("evaluation/results/generation_evaluation_report.md"),
        )
        print_summary_table(dense_summary)

    if args.mode in ("hybrid", "both"):
        hybrid_summary, hybrid_results = run_mode_eval("hybrid", provider, execution_isolation=isolation_mode)
        save_generation_results(
            hybrid_summary,
            hybrid_results,
            Path("evaluation/results/generation_evaluation_hybrid.json"),
            Path("evaluation/results/generation_evaluation_hybrid_report.md"),
        )
        print_summary_table(hybrid_summary)

    if args.mode == "both" and dense_summary and hybrid_summary:
        comp_report_path = Path("evaluation/results/generation_comparison_report.md")
        comp_json_path = Path("evaluation/results/generation_comparison.json")
        generate_comparison_report(dense_summary, hybrid_summary, comp_report_path, comp_json_path)
        print_comparison_table(dense_summary, hybrid_summary, comp_report_path, comp_json_path)


if __name__ == "__main__":
    main()
