"""Command-line interface for dense retrieval."""

import argparse
import sys
from pathlib import Path

from src.retrieval.models import RetrievalConfig
from src.retrieval.retriever import DenseRetriever


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments for dense retrieval."""
    parser = argparse.ArgumentParser(
        description="Search indexed research papers using dense vector retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Research question or search query string.",
    )
    parser.add_argument(
        "--query", "-q",
        dest="query_opt",
        type=str,
        default=None,
        help="Research question or search query string (alternative to positional argument).",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top scoring chunks to retrieve.",
    )
    parser.add_argument(
        "--doc-id", "-d",
        type=str,
        nargs="+",
        default=None,
        help="Filter results by one or more document IDs (e.g. -d P1 or -d P1 P2).",
    )
    parser.add_argument(
        "--collection-name", "-c",
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
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode: 'hybrid' (BM25 + Dense RRF), 'dense', or 'bm25'.",
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
        help="Reciprocal Rank Fusion smoothing constant.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for dense retrieval ranks in RRF.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for BM25 retrieval ranks in RRF.",
    )
    parser.add_argument(
        "--corpus-chunks",
        type=Path,
        default=Path("data/chunked/corpus_chunks.json"),
        help="Path to chunked corpus JSON file for BM25 indexing.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity / RRF score threshold.",
    )
    return parser.parse_args(args)


def main(args=None) -> int:
    """Execute retrieval CLI across dense, BM25, or hybrid modes."""
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
    query_text = parsed.query or parsed.query_opt

    if not query_text or not query_text.strip():
        print("Error: A search query must be provided via positional argument or --query/-q", file=sys.stderr)
        return 1

    mode = parsed.mode.lower()
    print("=" * 70)
    print(f" Research Intelligence Engine -- Retrieval ({mode.upper()} Mode)")
    print("=" * 70)
    print(f"Query:           {query_text}")
    print(f"Mode:            {mode}")
    print(f"Top-K:           {parsed.top_k}")
    print(f"Doc Filter:      {parsed.doc_id or 'None (All 5 Papers)'}")
    if mode in ("hybrid", "dense"):
        print(f"Collection:      {parsed.collection_name}")
        print(f"Qdrant Storage:  {parsed.qdrant_path.resolve()}")
    if mode in ("hybrid", "bm25"):
        print(f"Corpus Chunks:   {parsed.corpus_chunks.resolve()}")
    if mode == "hybrid":
        print(f"RRF Config:      k={parsed.rrf_k}, weights=(dense: {parsed.dense_weight}, bm25: {parsed.bm25_weight})")
        print(f"Candidate-K:     {parsed.candidate_k}")
    print("-" * 70)

    try:
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.models import BM25Config, HybridRetrievalConfig

        dense_cfg = RetrievalConfig(
            collection_name=parsed.collection_name,
            top_k=parsed.candidate_k if mode == "hybrid" else parsed.top_k,
            min_score=parsed.min_score if mode == "dense" else None,
            qdrant_path=parsed.qdrant_path,
            model_name=parsed.model_name,
        )
        bm25_cfg = BM25Config(
            corpus_chunks_path=parsed.corpus_chunks,
            top_k=parsed.candidate_k if mode == "hybrid" else parsed.top_k,
        )

        if mode == "dense":
            retriever_cm = DenseRetriever(config=dense_cfg)
        elif mode == "bm25":
            retriever_cm = BM25Retriever(config=bm25_cfg)
        else:
            hybrid_cfg = HybridRetrievalConfig(
                dense_config=dense_cfg,
                bm25_config=bm25_cfg,
                top_k=parsed.top_k,
                candidate_k=parsed.candidate_k,
                rrf_k=parsed.rrf_k,
                dense_weight=parsed.dense_weight,
                bm25_weight=parsed.bm25_weight,
            )
            retriever_cm = HybridRetriever(config=hybrid_cfg)

        with retriever_cm as retriever:
            result = retriever.retrieve(
                query=query_text,
                top_k=parsed.top_k,
                document_ids=parsed.doc_id,
                min_score=parsed.min_score,
            )

        print(f"Retrieved {result.total_retrieved} chunks in {result.execution_time_ms:.1f} ms\n")

        if not result.chunks:
            print("No matching chunks found above threshold.")
            print("=" * 70)
            return 0

        for rank, chunk in enumerate(result.chunks, 1):
            citation = chunk.formatted_citation()
            score_desc = f"Score: {chunk.score:.5f}"
            if chunk.retrieval_method == "hybrid_rrf":
                d_info = f"D-Rank: {chunk.dense_rank}" if chunk.dense_rank else "D-Rank: —"
                b_info = f"BM25-Rank: {chunk.bm25_rank}" if chunk.bm25_rank else "BM25-Rank: —"
                score_desc = f"RRF Score: {chunk.score:.6f} [{d_info}, {b_info}]"

            print(f"[{rank}] {score_desc} | Citation: {citation}")
            print(f"    Title: {chunk.document_title}")
            print(f"    Offsets: chars {chunk.start_char_idx}-{chunk.end_char_idx} ({chunk.char_count} chars, {chunk.word_count} words)")
            print("    Text Excerpt:")
            lines = chunk.text.strip().split("\n")
            preview = " ".join(line.strip() for line in lines if line.strip())
            if len(preview) > 350:
                preview = preview[:347] + "..."
            print(f"      \"{preview}\"")
            print("-" * 70)

        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] Retrieval failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

