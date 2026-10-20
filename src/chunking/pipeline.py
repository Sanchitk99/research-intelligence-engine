"""Pipeline for batch chunking of ingested document pages.

Loads ingested DocumentPage records, validates integrity, runs TextChunker,
aggregates statistics, and persists generated DocumentChunk records.
"""

import json
from pathlib import Path
from typing import Optional, Union, List

from src.ingestion.models import DocumentPage
from src.chunking.models import ChunkingConfig, DocumentChunk, ChunkingSummary
from src.chunking.text_chunker import TextChunker


class ChunkingPipelineError(Exception):
    """Raised when the chunking pipeline encounters invalid inputs or fatal errors."""
    pass


def chunk_corpus(
    input_path: Union[str, Path] = "data/ingested/corpus_pages.json",
    output_path: Optional[Union[str, Path]] = "data/chunked/corpus_chunks.json",
    config: Optional[ChunkingConfig] = None,
) -> ChunkingSummary:
    """Run text chunking on ingested document pages.

    Args:
        input_path: Path to ingested JSON file containing DocumentPage records.
        output_path: Destination JSON path to save generated DocumentChunk records.
        config: Optional ChunkingConfig overrides.

    Returns:
        ChunkingSummary containing processing metrics and per-document breakdown.
    """
    in_file = Path(input_path).resolve()
    if not in_file.is_file():
        raise FileNotFoundError(f"Ingested pages file not found: {in_file}")

    # 1. Load and validate input pages
    try:
        with open(in_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ChunkingPipelineError(f"Malformed JSON in ingested pages file: {e}") from e

    if not isinstance(raw_data, list):
        raise ChunkingPipelineError("Input file must contain a JSON array of DocumentPage objects.")

    pages: List[DocumentPage] = []
    failed_pages = 0
    empty_pages_count = 0

    for idx, item in enumerate(raw_data):
        try:
            page = DocumentPage.model_validate(item)
            pages.append(page)
            if page.page.is_empty or not page.text or not page.text.strip():
                empty_pages_count += 1
        except Exception:
            failed_pages += 1

    # 2. Execute chunking
    chunker = TextChunker(config=config)
    chunks = chunker.chunk_document_pages(pages)

    # 3. Aggregate statistics
    doc_ids = set()
    chunks_per_doc: dict[str, int] = {}
    total_chars = 0

    for chunk in chunks:
        doc_id = chunk.metadata.document_id
        doc_ids.add(doc_id)
        chunks_per_doc[doc_id] = chunks_per_doc.get(doc_id, 0) + 1
        total_chars += chunk.metadata.char_count

    avg_chars = round(total_chars / len(chunks), 2) if chunks else 0.0

    summary = ChunkingSummary(
        total_documents=len(doc_ids),
        total_pages_processed=len(pages),
        total_chunks_created=len(chunks),
        total_empty_pages_skipped=empty_pages_count,
        failed_pages=failed_pages,
        avg_chunk_chars=avg_chars,
        chunks_per_document=chunks_per_doc,
    )

    # 4. Save to output file if requested
    if output_path is not None:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = [chunk.model_dump() for chunk in chunks]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        summary.output_file = str(out_file)

    return summary

