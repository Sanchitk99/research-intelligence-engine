"""Command-line interface for running the retrieval evaluation benchmark."""

import argparse
import sys
from pathlib import Path

from src.retrieval.models import RetrievalConfig
from src.retrieval.retriever import DenseRetriever
from src.evaluation.ground_truth import load_ground_truth
from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.reporting import save_json_results, generate_markdown_report


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval quality against annotated evaluation questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("evaluation/questions/answerable_questions.json"),
        help="Path to annotated answerable questions JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluation/results/retrieval_evaluation.json"),
        help="Destination path for machine-readable evaluation results.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("evaluation/results/retrieval_evaluation_report.md"),
        help="Destination path for human-readable evaluation report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top-k cutoff for evaluation metrics.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="research_corpus",
        help="Target Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("data/qdrant"),
        help="Path to persistent local Qdrant directory.",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["both", "hybrid", "dense"],
        default="both",
        help="Evaluation mode: 'both' (runs Dense + Hybrid with side-by-side comparison), 'hybrid', or 'dense'.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--corpus-chunks",
        type=Path,
        default=Path("data/chunked/corpus_chunks.json"),
        help="Path to corpus_chunks.json for BM25.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Candidate pool size retrieved from each system before fusion.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Smoothing constant for Reciprocal Rank Fusion.",
    )
    return parser.parse_args(args)


def main(args=None) -> int:
    """Execute retrieval evaluation benchmark CLI."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parsed = parse_args(args)

    mode = parsed.mode.lower()
    print("=" * 72)
    print(f" Research Intelligence Engine -- Retrieval Evaluation Benchmark ({mode.upper()} Mode)")
    print("=" * 72)
    print(f"Questions File:  {parsed.questions_file.resolve()}")
    print(f"Mode:            {mode}")
    print(f"Evaluation Top-K:{parsed.top_k}")
    if mode in ("both", "dense"):
        print(f"Qdrant Storage:  {parsed.qdrant_path.resolve()} (Collection: {parsed.collection_name})")
        print(f"Embedding Model: {parsed.model_name}")
    if mode in ("both", "hybrid"):
        print(f"Corpus Chunks:   {parsed.corpus_chunks.resolve()}")
        print(f"RRF Config:      k={parsed.rrf_k}, candidate_k={parsed.candidate_k}")
    print("-" * 72)

    if not parsed.questions_file.is_file():
        print(f"Error: Questions file not found at {parsed.questions_file}", file=sys.stderr)
        return 1

    try:
        from src.evaluation.reporting import generate_comparison_markdown_report
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.models import BM25Config, HybridRetrievalConfig

        targets = load_ground_truth(parsed.questions_file)
        print(f"Loaded {len(targets)} ground-truth questions.")

        dense_cfg = RetrievalConfig(
            collection_name=parsed.collection_name,
            qdrant_path=parsed.qdrant_path,
            model_name=parsed.model_name,
            top_k=parsed.candidate_k if mode in ("both", "hybrid") else parsed.top_k,
        )
        bm25_cfg = BM25Config(
            corpus_chunks_path=parsed.corpus_chunks,
            top_k=parsed.candidate_k if mode in ("both", "hybrid") else parsed.top_k,
        )
        hybrid_cfg = HybridRetrievalConfig(
            dense_config=dense_cfg,
            bm25_config=bm25_cfg,
            top_k=parsed.top_k,
            candidate_k=parsed.candidate_k,
            rrf_k=parsed.rrf_k,
        )

        dense_summary, dense_results = None, None
        hybrid_summary, hybrid_results = None, None

        # 1. Run Dense Evaluation if required
        if mode in ("both", "dense"):
            print("\nEvaluating Dense Retrieval...")
            with DenseRetriever(config=dense_cfg) as d_retriever:
                d_evaluator = RetrievalEvaluator(retriever=d_retriever, default_top_k=parsed.top_k)
                dense_summary, dense_results = d_evaluator.evaluate_all(
                    targets=targets,
                    total_corpus_questions=len(targets),
                    excluded_count=10,
                    top_k=parsed.top_k,
                )
            dense_json = Path("evaluation/results/retrieval_evaluation.json")
            dense_report = Path("evaluation/results/retrieval_evaluation_report.md")
            save_json_results(dense_summary, dense_results, dense_json)
            generate_markdown_report(
                dense_summary,
                dense_results,
                dense_report,
                title="Retrieval Evaluation Baseline Report (Task 3.5 & Audit 3.5.1)",
                retriever_info="`DenseRetriever` | **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, Cosine)",
            )
            print(f"Dense Baseline saved -> {dense_report}")

        # 2. Run Hybrid Evaluation if required
        if mode in ("both", "hybrid"):
            print("\nEvaluating Hybrid Retrieval (BM25 + Dense RRF)...")
            with HybridRetriever(config=hybrid_cfg) as h_retriever:
                h_evaluator = RetrievalEvaluator(retriever=h_retriever, default_top_k=parsed.top_k)
                hybrid_summary, hybrid_results = h_evaluator.evaluate_all(
                    targets=targets,
                    total_corpus_questions=len(targets),
                    excluded_count=10,
                    top_k=parsed.top_k,
                )
            hybrid_json = Path("evaluation/results/hybrid_retrieval_evaluation.json")
            hybrid_report = Path("evaluation/results/hybrid_retrieval_evaluation_report.md")
            save_json_results(hybrid_summary, hybrid_results, hybrid_json)
            generate_markdown_report(
                hybrid_summary,
                hybrid_results,
                hybrid_report,
                title="Hybrid Retrieval Evaluation Report (Task 4.1)",
                retriever_info=f"`HybridRetriever` | BM25Okapi + `all-MiniLM-L6-v2` | RRF k={parsed.rrf_k}",
            )
            print(f"Hybrid Results saved -> {hybrid_report}")

        # 3. Generate side-by-side comparison if running both
        if mode == "both" and dense_summary and hybrid_summary and dense_results and hybrid_results:
            comparison_report = Path("evaluation/results/hybrid_vs_dense_comparison_report.md")
            generate_comparison_markdown_report(
                dense_summary=dense_summary,
                dense_results=dense_results,
                hybrid_summary=hybrid_summary,
                hybrid_results=hybrid_results,
                output_path=comparison_report,
            )
            print(f"Side-by-side Comparison saved -> {comparison_report}")

            print("\n" + "=" * 72)
            print(" RETRIEVAL COMPARISON SUMMARY (Task 4.1)")
            print("=" * 72)
            print(f"{'Metric':<30} | {'Dense Baseline':<16} | {'Hybrid (RRF)':<16} | {'Delta':<10}")
            print("-" * 72)
            print(f"{'Hit@1 (Page-Level Acc.)':<30} | {dense_summary.recall_at_1*100:>14.1f}% | {hybrid_summary.recall_at_1*100:>14.1f}% | {(hybrid_summary.recall_at_1 - dense_summary.recall_at_1)*100:>+8.1f}%")
            print(f"{'Hit@3 (Page-Level Acc.)':<30} | {dense_summary.recall_at_3*100:>14.1f}% | {hybrid_summary.recall_at_3*100:>14.1f}% | {(hybrid_summary.recall_at_3 - dense_summary.recall_at_3)*100:>+8.1f}%")
            print(f"{'Hit@5 (Page-Level Acc.)':<30} | {dense_summary.recall_at_5*100:>14.1f}% | {hybrid_summary.recall_at_5*100:>14.1f}% | {(hybrid_summary.recall_at_5 - dense_summary.recall_at_5)*100:>+8.1f}%")
            print(f"{'Mean Reciprocal Rank (MRR@5)':<30} | {dense_summary.mrr:>15.4f} | {hybrid_summary.mrr:>15.4f} | {(hybrid_summary.mrr - dense_summary.mrr):>+9.4f}")
            print(f"{'Mean Target Page Recall@5':<30} | {dense_summary.mean_target_page_recall_at_5*100:>14.1f}% | {hybrid_summary.mean_target_page_recall_at_5*100:>14.1f}% | {(hybrid_summary.mean_target_page_recall_at_5 - dense_summary.mean_target_page_recall_at_5)*100:>+8.1f}%")
            print(f"{'Mean Document Coverage@5':<30} | {dense_summary.mean_document_coverage_at_5*100:>14.1f}% | {hybrid_summary.mean_document_coverage_at_5*100:>14.1f}% | {(hybrid_summary.mean_document_coverage_at_5 - dense_summary.mean_document_coverage_at_5)*100:>+8.1f}%")
            print("=" * 72)

        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] Evaluation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

