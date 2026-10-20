"""Command-line interface for text chunking.

Usage:
    python -m src.chunking.cli
    python -m src.chunking.cli --input data/ingested/corpus_pages.json --output data/chunked/corpus_chunks.json --chunk-size 1000 --chunk-overlap 150
"""

import sys
import argparse
from pathlib import Path

from src.chunking.models import ChunkingConfig
from src.chunking.pipeline import chunk_corpus


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for chunking CLI."""
    parser = argparse.ArgumentParser(
        description="Chunk ingested PDF page records into retrieval-sized text segments."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/ingested/corpus_pages.json",
        help="Input JSON file containing DocumentPage records (default: data/ingested/corpus_pages.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chunked/corpus_chunks.json",
        help="Output JSON file for generated DocumentChunk records (default: data/chunked/corpus_chunks.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Target overlap between neighboring chunks in characters (default: 150)",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=50,
        help="Minimum standalone chunk size in characters (default: 50)",
    )
    return parser


def main() -> int:
    """Execute chunking CLI."""
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path.cwd()
    input_file = (project_root / args.input).resolve()
    output_file = (project_root / args.output).resolve()

    print("=================================================================")
    print(" Research Intelligence Engine -- Text Chunking Module (Task 3.2)")
    print("=================================================================")
    print(f"Input Pages:   {input_file}")
    print(f"Output Chunks: {output_file}")
    print(f"Settings:      ChunkSize={args.chunk_size} chars | Overlap={args.chunk_overlap} chars\n")

    if not input_file.is_file():
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        return 1

    try:
        config = ChunkingConfig(
            chunk_size_chars=args.chunk_size,
            chunk_overlap_chars=args.chunk_overlap,
            min_chunk_size_chars=args.min_chunk_size,
        )
    except ValueError as e:
        print(f"ERROR: Invalid chunking configuration: {e}", file=sys.stderr)
        return 1

    try:
        summary = chunk_corpus(
            input_path=input_file,
            output_path=output_file,
            config=config,
        )
    except Exception as e:
        print(f"FATAL: Chunking failed with exception: {e}", file=sys.stderr)
        return 1

    print("-----------------------------------------------------------------")
    print(f"Documents Processed:    {summary.total_documents}")
    print(f"Pages Processed:        {summary.total_pages_processed}")
    print(f"Empty Pages Skipped:    {summary.total_empty_pages_skipped}")
    print(f"Failed Pages:           {summary.failed_pages}")
    print(f"Total Chunks Created:   {summary.total_chunks_created}")
    print(f"Average Chunk Length:   {summary.avg_chunk_chars} chars")
    print("-----------------------------------------------------------------")
    print(f"{'Doc ID':<8} | {'Chunks':<8} | {'Status'}")
    print("-" * 35)

    for doc_id, count in sorted(summary.chunks_per_document.items()):
        print(f"{doc_id:<8} | {count:<8} | OK")

    print("-----------------------------------------------------------------")
    if summary.output_file and Path(summary.output_file).is_file():
        size_kb = Path(summary.output_file).stat().st_size / 1024
        print(f"Saved {summary.total_chunks_created} chunk records to:")
        print(f"  {summary.output_file} ({size_kb:.1f} KB)")
    print("=================================================================\n")

    return 0 if summary.failed_pages == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

