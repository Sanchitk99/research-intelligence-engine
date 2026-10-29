"""Data models and schemas for dense retrieval in the Research Intelligence Engine."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RetrievalError(Exception):
    """Base exception for all retrieval operations."""
    pass


class InvalidQueryError(RetrievalError):
    """Raised when a query string is empty, whitespace-only, or invalid."""
    pass


class CollectionNotFoundError(RetrievalError):
    """Raised when the specified Qdrant collection does not exist."""
    pass


class FilterError(RetrievalError):
    """Raised when document filter criteria are invalid."""
    pass


class RetrievalConfig(BaseModel):
    """Configuration settings for dense retrieval."""
    model_config = ConfigDict(extra="ignore")

    collection_name: str = Field(
        default="research_corpus",
        description="Name of the Qdrant collection to search"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top scoring chunks to retrieve"
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity threshold"
    )
    qdrant_path: Path = Field(
        default=Path("data/qdrant"),
        description="Persistent filesystem path for local Qdrant database"
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name used for query embedding"
    )


class BM25Config(BaseModel):
    """Configuration settings for BM25 keyword retrieval."""
    model_config = ConfigDict(extra="ignore")

    corpus_chunks_path: Path = Field(
        default=Path("data/chunked/corpus_chunks.json"),
        description="Path to chunked corpus JSON file"
    )
    k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 term frequency saturation parameter"
    )
    b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 document length normalization parameter"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top scoring chunks to retrieve"
    )


class HybridRetrievalConfig(BaseModel):
    """Configuration settings for hybrid (Dense + BM25) retrieval with RRF."""
    model_config = ConfigDict(extra="ignore")

    dense_config: Optional[RetrievalConfig] = Field(
        default=None,
        description="Dense retrieval configuration"
    )
    bm25_config: Optional[BM25Config] = Field(
        default=None,
        description="BM25 keyword retrieval configuration"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Final number of fused chunks to return"
    )
    candidate_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Candidate pool size retrieved from each subsystem before fusion"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Smoothing constant for Reciprocal Rank Fusion"
    )
    dense_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight multiplier for dense retrieval ranks in RRF"
    )
    bm25_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight multiplier for BM25 retrieval ranks in RRF"
    )
    max_chunks_per_doc: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum chunks allowed per document for diversification (None disables diversification)"
    )
    enable_intent_routing: bool = Field(
        default=False,
        description="Whether to dynamically route retrieval configuration based on query intent (opt-in)"
    )
    comparative_max_chunks_per_doc: int = Field(
        default=2,
        ge=1,
        description="Per-document chunk cap to apply when query intent is COMPARATIVE"
    )


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with verbatim text, similarity score, and full citation provenance."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(description="Unique deterministic chunk ID (e.g. 'P1_p01_c01')")
    document_id: str = Field(description="Source document identifier (e.g. 'P1')")
    document_title: str = Field(description="Full document title")
    physical_page_index: int = Field(description="Zero-based physical page index in PDF")
    physical_page_number: int = Field(description="One-based physical page number in PDF")
    printed_page_number: Optional[int] = Field(
        default=None,
        description="Printed page number if verified from document/mapping; None for preprints"
    )
    chunk_index_in_page: int = Field(description="Zero-based chunk index on the page")
    chunk_index_in_doc: int = Field(description="Zero-based chunk index in the document")
    start_char_idx: int = Field(description="Zero-based start character offset in page text")
    end_char_idx: int = Field(description="Zero-based end character offset in page text")
    char_count: int = Field(description="Character length of verbatim chunk text")
    word_count: int = Field(description="Word count of verbatim chunk text")
    text: str = Field(description="Verbatim chunk text extracted from source page")
    score: float = Field(description="Cosine similarity score or RRF fusion score")
    retrieval_method: Optional[str] = Field(
        default=None,
        description="Method used to retrieve chunk: 'dense', 'bm25', or 'hybrid_rrf'"
    )
    dense_rank: Optional[int] = Field(
        default=None,
        description="1-based rank in dense retrieval if present in candidate pool"
    )
    bm25_rank: Optional[int] = Field(
        default=None,
        description="1-based rank in BM25 retrieval if present in candidate pool"
    )
    dense_score: Optional[float] = Field(
        default=None,
        description="Raw cosine similarity score from dense search"
    )
    bm25_score: Optional[float] = Field(
        default=None,
        description="Raw BM25 score from keyword search"
    )

    def formatted_citation(self) -> str:
        """Generate a human-readable, precise citation string."""
        if self.printed_page_number is not None:
            page_info = f"Physical Page {self.physical_page_number} (Printed Page {self.printed_page_number})"
        else:
            page_info = f"Physical Page {self.physical_page_number} (Preprint)"
        return f"[{self.document_id}, {page_info}, Chunk {self.chunk_id}]"


class RetrievalResult(BaseModel):
    """Structured result set returned by dense retrieval for a query."""
    model_config = ConfigDict(extra="ignore")

    query: str = Field(description="Original user query text")
    chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Ranked list of retrieved chunks ordered by descending similarity"
    )
    total_retrieved: int = Field(description="Number of chunks retrieved")
    collection_name: str = Field(description="Qdrant collection searched")
    document_filter: Optional[List[str]] = Field(
        default=None,
        description="Document IDs used to restrict retrieval, or None if unfiltered"
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Total query embedding and search execution time in milliseconds"
    )
    dense_time_ms: Optional[float] = Field(
        default=None,
        description="Dense retrieval execution time in milliseconds"
    )
    bm25_time_ms: Optional[float] = Field(
        default=None,
        description="BM25 keyword retrieval execution time in milliseconds"
    )
    query_intent: Optional[str] = Field(
        default=None,
        description="Classified query intent ('COMPARATIVE' or 'STANDARD') if intent routing was enabled"
    )

