"""PDF Ingestion Module for Research Intelligence Engine."""

from src.ingestion.models import (
    DocumentMetadata,
    PageMetadata,
    DocumentPage,
    DocumentIngestionResult,
    BatchIngestionSummary,
)
from src.ingestion.loader import PDFLoader
from src.ingestion.pagination import PaginationResolver
from src.ingestion.registry import load_paper_registry
from src.ingestion.batch import batch_ingest

__all__ = [
    "DocumentMetadata",
    "PageMetadata",
    "DocumentPage",
    "DocumentIngestionResult",
    "BatchIngestionSummary",
    "PDFLoader",
    "PaginationResolver",
    "load_paper_registry",
    "batch_ingest",
]

