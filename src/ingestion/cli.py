"""Command-line interface for running batch PDF ingestion.

Usage:
    python -m src.ingestion.cli
    python -m src.ingestion.cli --registry path/to/registry.json --pdf-dir path/to/pdfs --output data/ingested/pages.json
"""

import sys
import argparse
from pathlib import Path
from src.ingestion.batch import batch_ingest


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest research paper PDFs, extract text and page metadata, and save page records."
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="evaluation/corpus/metadata/paper_registry.json",
        help="Path to paper_registry.json (default: evaluation/corpus/metadata/paper_registry.json)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="evaluation/corpus/papers",
        help="Directory containing the full-text PDFs (default: evaluation/corpus/papers)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ingested/corpus_pages.json",
        help="Output JSON file for extracted DocumentPage records (default: data/ingested/corpus_pages.json)",
    )
    return parser


def main() -> int:
    """Execute ingestion CLI."""
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path.cwd()
    registry_path = (project_root / args.registry).resolve()
    pdf_dir = (project_root / args.pdf_dir).resolve()
    output_path = (project_root / args.output).resolve()

    print("=================================================================")
    print(" Research Intelligence Engine -- PDF Ingestion Module (Phase 2)")
    print("=================================================================")
    print(f"Registry: {registry_path}")
    print(f"PDF Dir:  {pdf_dir}")
    print(f"Output:   {output_path}\n")

    if not registry_path.is_file():
        print(f"ERROR: Registry file not found: {registry_path}", file=sys.stderr)
        return 1

    if not pdf_dir.is_dir():
        print(f"ERROR: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    try:
        summary = batch_ingest(
            registry_path=registry_path,
            pdf_dir=pdf_dir,
            output_path=output_path,
        )
    except Exception as e:
        print(f"FATAL: Ingestion failed with exception: {e}", file=sys.stderr)
        return 1

    print("-----------------------------------------------------------------")
    print(f"Documents Registered:  {summary.total_documents}")
    print(f"Successfully Ingested: {summary.successful_documents}")
    print(f"Extraction Failures:   {summary.failed_documents}")
    print(f"Total Pages Extracted: {summary.total_pages_extracted}")
    print(f"Total Empty Pages:     {summary.total_empty_pages}")
    print("-----------------------------------------------------------------")
    print(f"{'ID':<6} | {'Status':<7} | {'Pages':<6} | {'Empty':<6} | {'Title'}")
    print("-" * 65)

    for res in summary.results:
        status_str = "OK" if res.success else "FAILED"
        first_page_title = res.pages[0].document.title if res.pages else res.document_id
        short_title = (first_page_title[:38] + "..") if len(first_page_title) > 40 else first_page_title
        print(f"{res.document_id:<6} | {status_str:<7} | {res.total_pages:<6} | {res.empty_pages_count:<6} | {short_title}")
        if not res.success:
            print(f"       -> Error: {res.error_message}")

    print("-----------------------------------------------------------------")
    if summary.output_file and Path(summary.output_file).is_file():
        size_kb = Path(summary.output_file).stat().st_size / 1024
        print(f"Saved {summary.total_pages_extracted} page records to:")
        print(f"  {summary.output_file} ({size_kb:.1f} KB)")
    print("=================================================================\n")

    return 0 if summary.failed_documents == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
