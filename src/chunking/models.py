"""Data models for text chunking in the Research Intelligence Engine.

Maintains strict separation between chunk text and source provenance metadata.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ChunkingConfig(BaseModel):
    """Configuration settings for text chunking."""
    model_config = ConfigDict(extra="ignore")

    chunk_size_chars: int = Field(
        default=1000,
        ge=50,
        description="Target maximum chunk size in characters"
    )
    chunk_overlap_chars: int = Field(
        default=150,
        ge=0,
        description="Target character overlap between neighboring chunks"
    )
    min_chunk_size_chars: int = Field(
        default=50,
        ge=1,
        description="Minimum character size for a standalone chunk"
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError(
                f"chunk_overlap_chars ({self.chunk_overlap_chars}) must be less than "
                f"chunk_size_chars ({self.chunk_size_chars})"
            )
        return self


class ChunkMetadata(BaseModel):
    """Provenance and positional metadata for an individual chunk."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(description="Deterministic chunk ID (e.g. 'P1_p01_c01')")
    document_id: str = Field(description="Source document identifier (e.g. 'P1')")
    document_title: str = Field(description="Source document title")
    physical_page_index: int = Field(description="Zero-based physical page index in PDF")
    physical_page_number: int = Field(description="One-based physical page number in PDF")
    printed_page_number: Optional[int] = Field(
        default=None,
        description="Printed page number if verified from document/mapping; None for preprints"
    )
    chunk_index_in_page: int = Field(description="Zero-based chunk index within the page")
    chunk_index_in_doc: int = Field(description="Zero-based chunk index within the document")
    start_char_idx: int = Field(description="Zero-based start character offset in page text")
    end_char_idx: int = Field(description="Zero-based end character offset in page text")
    char_count: int = Field(description="Character length of verbatim chunk text")
    word_count: int = Field(description="Word count of verbatim chunk text")


class DocumentChunk(BaseModel):
    """A single text chunk with decoupled provenance metadata and verbatim text."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(description="Deterministic chunk ID")
    metadata: ChunkMetadata = Field(description="Provenance and offset metadata")
    text: str = Field(description="Verbatim chunk text extracted from source page")


class ChunkingSummary(BaseModel):
    """Summary report for a batch chunking run."""
    model_config = ConfigDict(extra="ignore")

    total_documents: int = Field(default=0, description="Total unique documents processed")
    total_pages_processed: int = Field(default=0, description="Total pages processed")
    total_chunks_created: int = Field(default=0, description="Total chunks generated")
    total_empty_pages_skipped: int = Field(default=0, description="Empty or whitespace pages skipped")
    failed_pages: int = Field(default=0, description="Pages that failed chunking")
    avg_chunk_chars: float = Field(default=0.0, description="Average chunk length in characters")
    chunks_per_document: Dict[str, int] = Field(
        default_factory=dict,
        description="Chunk counts broken down by document ID"
    )
    output_file: Optional[str] = Field(default=None, description="Path where chunks were saved")

