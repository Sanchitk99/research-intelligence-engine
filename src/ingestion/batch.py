"""Batch ingestion pipeline for registered research papers.

Resolves paths relative to the project root, executes safe page extraction,
aggregates statistics, and persists extracted page records to JSON.
"""

import json
from pathlib import Path
from typing import Optional, Union, List
from src.ingestion.models import (
    DocumentPage,
    DocumentIngestionResult,
    BatchIngestionSummary,
)
from src.ingestion.loader import PDFLoader
from src.ingestion.registry import load_paper_registry


def batch_ingest(
    registry_path: Union[str, Path],
    pdf_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> BatchIngestionSummary:
    """Ingest all PDFs registered in the paper registry.

    Args:
        registry_path: Path to paper_registry.json.
        pdf_dir: Directory containing the PDF files.
        output_path: Optional destination path to save extracted DocumentPage records.

    Returns:
        BatchIngestionSummary containing execution metrics and per-document results.
    """
    registry_file = Path(registry_path).resolve()
    pdf_directory = Path(pdf_dir).resolve()

    # 1. Load registered papers
    papers = load_paper_registry(registry_file)

    loader = PDFLoader(raise_on_error=False)
    results: List[DocumentIngestionResult] = []
    all_extracted_pages: List[DocumentPage] = []

    successful_docs = 0
    failed_docs = 0
    total_pages = 0
    total_empty = 0

    # 2. Ingest each paper
    for paper in papers:
        pdf_file = pdf_directory / (paper.filename or f"{paper.id}.pdf")

        result = loader.load_document(
            file_path=pdf_file,
            metadata=paper,
        )
        results.append(result)

        if result.success:
            successful_docs += 1
            total_pages += result.total_pages
            total_empty += result.empty_pages_count
            all_extracted_pages.extend(result.pages)
        else:
            failed_docs += 1

    summary = BatchIngestionSummary(
        total_documents=len(papers),
        successful_documents=successful_docs,
        failed_documents=failed_docs,
        total_pages_extracted=total_pages,
        total_empty_pages=total_empty,
        results=results,
    )

    # 3. Persist extracted pages to predictable output file if requested
    if output_path is not None:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        pages_payload = [p.model_dump() for p in all_extracted_pages]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(pages_payload, f, indent=2, ensure_ascii=False)

        summary.output_file = str(out_file)

    return summary

