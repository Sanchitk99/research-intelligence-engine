"""Data models for PDF ingestion in the Research Intelligence Engine.

Separates document-level identity and provenance metadata from page-level
numbering and raw extracted text.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class DocumentMetadata(BaseModel):
    """Metadata describing the document source and publication identity."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Unique paper identifier, e.g. 'P1'")
    title: str = Field(description="Full official paper title")
    short_name: Optional[str] = Field(default=None, description="Short canonical name, e.g. 'RAG'")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Publication venue or conference")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. '2005.11401'")
    doi: Optional[str] = Field(default=None, description="DOI or official proceedings URL")
    pdf_url: Optional[str] = Field(default=None, description="Direct URL to full-text PDF")
    filename: Optional[str] = Field(default=None, description="Local PDF filename in papers dir")
    total_pages: Optional[int] = Field(default=None, description="Expected total page count")
    pagination_mapping: Optional[str] = Field(default=None, description="Verified pagination formula/rule")
    downloaded: Optional[bool] = Field(default=None, description="Local availability status")
    pages_verified: Optional[bool] = Field(default=None, description="Page verification status")


class PageMetadata(BaseModel):
    """Page-level location and layout metadata.
    
    Distinguishes the zero-based physical PDF page index from the one-based
    physical page number, and records the printed page number ONLY when grounded
    in a verified mapping or document evidence.
    """
    model_config = ConfigDict(extra="ignore")

    physical_page_index: int = Field(
        description="Zero-based physical page index in the PDF document (0, 1, 2, ...)"
    )
    physical_page_number: int = Field(
        description="One-based physical page number in the PDF file (1, 2, 3, ...)"
    )
    printed_page_number: Optional[int] = Field(
        default=None,
        description="Printed page number if verified from document/mapping; None for preprints/unverified"
    )
    is_empty: bool = Field(
        default=False,
        description="True if the extracted text contains no characters or only whitespace"
    )
    char_count: int = Field(
        default=0,
        description="Total character count of the verbatim extracted page text"
    )
    word_count: int = Field(
        default=0,
        description="Total word count of the verbatim extracted page text"
    )


class DocumentPage(BaseModel):
    """A single extracted page with decoupled document metadata and verbatim text."""
    model_config = ConfigDict(extra="ignore")

    document: DocumentMetadata = Field(description="Document identity and source metadata")
    page: PageMetadata = Field(description="Page numbering and structural flags")
    text: str = Field(description="Verbatim extracted text from the page without rewriting")


class DocumentIngestionResult(BaseModel):
    """Result of ingesting a single PDF document."""
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(description="Paper ID of the ingested document")
    file_path: str = Field(description="Path to the source PDF file")
    success: bool = Field(description="True if ingestion completed without fatal errors")
    total_pages: int = Field(default=0, description="Total pages processed")
    empty_pages_count: int = Field(default=0, description="Number of empty or whitespace-only pages")
    pages: List[DocumentPage] = Field(default_factory=list, description="List of extracted DocumentPage records")
    error_message: Optional[str] = Field(default=None, description="Error description if ingestion failed")


class BatchIngestionSummary(BaseModel):
    """Aggregate summary report for a batch ingestion run."""
    model_config = ConfigDict(extra="ignore")

    total_documents: int = Field(default=0, description="Total documents registered in batch")
    successful_documents: int = Field(default=0, description="Documents successfully ingested")
    failed_documents: int = Field(default=0, description="Documents that failed ingestion")
    total_pages_extracted: int = Field(default=0, description="Total pages extracted across all documents")
    total_empty_pages: int = Field(default=0, description="Total empty pages encountered")
    results: List[DocumentIngestionResult] = Field(default_factory=list, description="Per-document ingestion results")
    output_file: Optional[str] = Field(default=None, description="Path where extracted pages were saved")

