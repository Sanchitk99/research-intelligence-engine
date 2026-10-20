"""Command-line interface for the indexing module."""

import argparse
import sys
from pathlib import Path

from src.indexing.models import IndexingConfig
from src.indexing.pipeline import IndexingPipeline


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments for corpus indexing."""
    parser = argparse.ArgumentParser(
        description="Index chunked research documents into local Qdrant vector storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/chunked/corpus_chunks.json"),
        help="Path to chunked documents JSON artifact.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("data/qdrant"),
        help="Local filesystem directory for persistent Qdrant database.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="research_corpus",
        help="Target Qdrant collection name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding inference and Qdrant upsert.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation and embedding without persisting points to Qdrant.",
    )
    return parser.parse_args(args)


def main(args=None) -> int:
    """Execute corpus indexing CLI."""
    parsed = parse_args(args)

    print("=" * 65)
    print(" Research Intelligence Engine -- Embedding & Vector Storage (Task 3.3)")
    print("=" * 65)
    print(f"Chunks File:     {parsed.chunks_file.resolve()}")
    print(f"Qdrant Storage:  {parsed.qdrant_path.resolve()}")
    print(f"Collection:      {parsed.collection_name}")
    print(f"Embedding Model: {parsed.model_name}")
    print(f"Batch Size:      {parsed.batch_size}")
    print(f"Dry Run:         {parsed.dry_run}")
    print("-" * 65)

    if not parsed.chunks_file.is_file():
        print(f"Error: Chunks file not found: {parsed.chunks_file}", file=sys.stderr)
        return 1

    config = IndexingConfig(
        collection_name=parsed.collection_name,
        model_name=parsed.model_name,
        qdrant_path=parsed.qdrant_path,
        batch_size=parsed.batch_size,
    )

    try:
        pipeline = IndexingPipeline(config=config)
        summary = pipeline.run(chunks_file=parsed.chunks_file, dry_run=parsed.dry_run)

        print("-----------------------------------------------------------------")
        print(f"Chunks Read:            {summary.chunks_read}")
        print(f"Chunks Embedded:        {summary.chunks_embedded}")
        print(f"Chunks Indexed:         {summary.chunks_indexed}")
        print(f"Chunks Failed/Skipped:  {summary.chunks_failed}")
        print(f"Model Name:             {summary.model_name}")
        print(f"Vector Dimension:       {summary.vector_dim}")
        print(f"Collection Name:        {summary.collection_name}")
        print(f"Total Collection Points:{summary.total_points_in_collection}")
        print(f"Execution Time:         {summary.execution_time_seconds:.2f} seconds")
        print("-----------------------------------------------------------------")
        if summary.dry_run:
            print("[DRY RUN COMPLETE] Chunks and embeddings validated successfully.")
        else:
            print(f"[INDEXING COMPLETE] Collection '{summary.collection_name}' ready in {parsed.qdrant_path}.")
        print("=" * 65)
        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] Indexing failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

