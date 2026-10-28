"""Paper library and document ingestion endpoints."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from src.api.schemas import (
    PaperSummaryResponse,
    CorpusPDFResponse,
    IngestPaperRequest,
    IngestionResponse,
)
from src.api.dependencies import get_research_service, get_project_root
from src.ui.service import ResearchEngineService, IngestionReport

logger = logging.getLogger(__name__)

# Canonical titles mapping for benchmark documents
KNOWN_TITLES = {
    "P1": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "P2": "Dense Passage Retrieval for Open-Domain Question Answering",
    "P3": "REALM: Retrieval-Augmented Language Model Pre-Training",
    "P4": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    "P5": "Ragas: Automated Evaluation of Retrieval Augmented Generation",
}

router = APIRouter(prefix="/api/papers", tags=["Papers"])


@router.get("", response_model=List[PaperSummaryResponse])
def get_indexed_papers(
    service: ResearchEngineService = Depends(get_research_service),
) -> List[PaperSummaryResponse]:
    """Retrieve all documents currently indexed in the active research corpus."""
    summaries = service.get_indexed_papers()
    return [
        PaperSummaryResponse(
            document_id=s.document_id,
            title=s.title,
            total_pages=s.total_pages,
            total_chunks=s.total_chunks,
        )
        for s in summaries
    ]


@router.get("/corpus", response_model=List[CorpusPDFResponse])
def list_available_corpus_pdfs() -> List[CorpusPDFResponse]:
    """List benchmark PDFs available in evaluation/corpus/papers/ for demonstration."""
    root = get_project_root()
    corpus_dir = root / "evaluation" / "corpus" / "papers"
    if not corpus_dir.is_dir():
        return []

    results = []
    for f in sorted(corpus_dir.glob("*.pdf")):
        # Extract doc_id prefix (e.g. 'P1')
        prefix = f.name.split("_")[0].upper()
        title = KNOWN_TITLES.get(prefix, f.stem.replace("_", " "))
        size = f.stat().st_size
        results.append(
            CorpusPDFResponse(
                filename=f.name,
                document_id=prefix,
                title=title,
                file_size_bytes=size,
            )
        )
    return results


@router.post("/ingest", response_model=IngestionResponse)
def ingest_corpus_paper(
    req: IngestPaperRequest,
    service: ResearchEngineService = Depends(get_research_service),
) -> IngestionResponse:
    """Ingest or dry-run index a document from the evaluation corpus."""
    root = get_project_root()
    corpus_dir = root / "evaluation" / "corpus" / "papers"
    target_path = corpus_dir / req.filename

    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus PDF '{req.filename}' was not found in evaluation/corpus/papers/.",
        )

    resolved_id = req.doc_id
    if not resolved_id and "_" in target_path.name:
        candidate = target_path.name.split("_")[0].upper()
        if candidate.startswith("P") and candidate[1:].isdigit():
            resolved_id = candidate

    resolved_title = req.doc_title or KNOWN_TITLES.get(resolved_id)

    try:
        report: IngestionReport = service.ingest_paper(
            pdf_path=target_path,
            doc_id=resolved_id,
            doc_title=resolved_title,
            dry_run=req.dry_run,
        )
    except Exception as e:
        logger.error(f"Ingestion failed for {req.filename}: {e}", exc_info=True)
        return IngestionResponse(
            success=False,
            document_id=req.doc_id or "UNKNOWN",
            title=req.doc_title or req.filename,
            filename=req.filename,
            pages_extracted=0,
            chunks_created=0,
            chunks_indexed=0,
            total_collection_points=0,
            error_message=str(e),
            sample_chunks=[],
        )

    return IngestionResponse(
        success=report.success,
        document_id=report.document_id,
        title=report.title,
        filename=report.filename,
        pages_extracted=report.pages_extracted,
        chunks_created=report.chunks_created,
        chunks_indexed=report.chunks_indexed,
        total_collection_points=report.total_collection_points,
        error_message=report.error_message,
        sample_chunks=report.sample_chunks,
        dry_run=report.dry_run,
    )


@router.post("/upload", response_model=IngestionResponse)
async def upload_and_ingest_pdf(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    doc_title: Optional[str] = Form(None),
    dry_run: bool = Form(False),
    service: ResearchEngineService = Depends(get_research_service),
) -> IngestionResponse:
    """Accept an uploaded PDF file, validate, and execute ingestion with dry-run support."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .pdf extension.",
        )

    # Save to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / file.filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate file size (max 25MB)
        if temp_file_path.stat().st_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds maximum permitted limit of 25MB.",
            )

        report: IngestionReport = service.ingest_paper(
            pdf_path=temp_file_path,
            doc_id=doc_id,
            doc_title=doc_title,
            dry_run=dry_run,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting uploaded PDF: {e}", exc_info=True)
        return IngestionResponse(
            success=False,
            document_id=doc_id or "UNKNOWN",
            title=doc_title or file.filename,
            filename=file.filename,
            pages_extracted=0,
            chunks_created=0,
            chunks_indexed=0,
            total_collection_points=0,
            error_message=str(e),
            sample_chunks=[],
        )
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    return IngestionResponse(
        success=report.success,
        document_id=report.document_id,
        title=report.title,
        filename=report.filename,
        pages_extracted=report.pages_extracted,
        chunks_created=report.chunks_created,
        chunks_indexed=report.chunks_indexed,
        total_collection_points=report.total_collection_points,
        error_message=report.error_message,
        sample_chunks=report.sample_chunks,
        dry_run=report.dry_run,
    )
