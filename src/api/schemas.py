"""Pydantic request and response schemas for the Research Intelligence Engine API."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    """Health status response."""
    status: str = Field(default="ok", description="Service health indicator")
    service: str = Field(default="Research Intelligence Engine API", description="Service title")
    version: str = Field(default="1.0.0", description="API version")


class ProviderStatus(BaseModel):
    """Status details for a single LLM provider."""
    configured: bool = Field(description="Whether provider credentials/environment are configured")
    default_model: str = Field(description="Default model string for this provider")


class PaperSummaryResponse(BaseModel):
    """Summary of an indexed research paper."""
    document_id: str = Field(description="Document ID (e.g. 'P1')")
    title: str = Field(description="Full paper title")
    total_pages: int = Field(description="Number of physical pages in document")
    total_chunks: int = Field(description="Number of chunks created and indexed")


class SystemStatusResponse(BaseModel):
    """Complete system health, collection metrics, and provider availability."""
    collection_name: str = Field(description="Active Qdrant collection name")
    total_points: int = Field(description="Total vector points in collection")
    indexed_papers_count: int = Field(description="Number of distinct indexed papers")
    papers: List[PaperSummaryResponse] = Field(default_factory=list, description="Indexed paper records")
    providers: Dict[str, ProviderStatus] = Field(description="Map of provider name to status")


class CitationResponse(BaseModel):
    """Verified citation item grounded in retrieved context."""
    chunk_id: str = Field(description="Chunk ID (e.g. 'P1_p04_c01')")
    document_id: str = Field(description="Source paper ID (e.g. 'P1')")
    document_title: str = Field(description="Full title of cited document")
    physical_page_number: int = Field(description="One-based physical page number")
    printed_page_number: Optional[int] = Field(default=None, description="Printed page number if mapped")
    excerpt: str = Field(description="Verbatim supporting text from source chunk")


class RetrievedChunkResponse(BaseModel):
    """Retrieved passage with similarity/RRF score and page provenance."""
    chunk_id: str = Field(description="Chunk ID")
    document_id: str = Field(description="Document ID")
    document_title: str = Field(description="Document title")
    physical_page_number: int = Field(description="Physical page number")
    printed_page_number: Optional[int] = Field(default=None, description="Printed page number if mapped")
    text: str = Field(description="Passage text")
    score: float = Field(description="Similarity or RRF fusion score")


class AskQuestionRequest(BaseModel):
    """User question payload with retrieval and provider options."""
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1, max_length=2000, description="Natural language research question")
    provider: str = Field(default="auto", description="LLM provider: 'auto', 'mock', 'gemini', 'openai', 'groq', 'ollama'")
    model: Optional[str] = Field(default=None, description="Optional model identifier override")
    retrieval_mode: str = Field(default="hybrid", description="Retrieval algorithm: 'hybrid' or 'dense'")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages to retrieve")
    min_score: Optional[float] = Field(default=None, description="Optional minimum score cutoff")
    enable_intent_routing: bool = Field(default=False, description="Opt-in query-intent routing for comparative queries")


class AskQuestionResponse(BaseModel):
    """Synthesized research answer with verified page citations and context."""
    question: str = Field(description="Original research question")
    answer: str = Field(description="Synthesized evidence-grounded answer or abstention message")
    status: str = Field(description="AnswerStatus value (e.g. 'answered', 'abstained_out_of_scope')")
    citations: List[CitationResponse] = Field(default_factory=list, description="Verified citations list")
    retrieved_chunks: List[RetrievedChunkResponse] = Field(default_factory=list, description="Retrieved context passages")
    model_name: str = Field(description="Model used for synthesis")
    provider_name: str = Field(description="Provider backend used")
    retrieval_mode: str = Field(description="Retrieval mode applied ('hybrid' or 'dense')")
    latency_ms: float = Field(description="Total end-to-end latency in milliseconds")
    confidence_notes: Optional[str] = Field(default=None, description="Synthesizer confidence notes")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt token count if reported")
    completion_tokens: Optional[int] = Field(default=None, description="Completion token count if reported")
    total_tokens: Optional[int] = Field(default=None, description="Total token count if reported")


class CorpusPDFResponse(BaseModel):
    """A research PDF available in the benchmark evaluation corpus."""
    filename: str = Field(description="Filename on disk (e.g. 'P5_RAGAS_Es_2023.pdf')")
    document_id: str = Field(description="Inferred or canonical document ID (e.g. 'P5')")
    title: str = Field(description="Human-readable paper title")
    file_size_bytes: int = Field(description="File size on disk")


class IngestPaperRequest(BaseModel):
    """Payload to trigger ingestion of a paper already in the corpus."""
    filename: str = Field(description="Target PDF filename in corpus")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID override")
    doc_title: Optional[str] = Field(default=None, description="Optional document title override")
    dry_run: bool = Field(default=False, description="Dry run without persisting to vector store")


class IngestionResponse(BaseModel):
    """Detailed report from the document ingestion and chunking pipeline."""
    success: bool = Field(description="Whether ingestion succeeded")
    document_id: str = Field(description="Assigned document identifier")
    title: str = Field(description="Document title")
    filename: str = Field(description="Source PDF filename")
    pages_extracted: int = Field(description="Number of physical pages extracted")
    chunks_created: int = Field(description="Number of chunks created")
    chunks_indexed: int = Field(description="Number of chunks indexed into vector store")
    total_collection_points: int = Field(description="New total points in collection")
    error_message: Optional[str] = Field(default=None, description="Error details if unsuccessful")
    sample_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Sample chunk previews")
    dry_run: bool = Field(default=False, description="Whether execution was dry-run only")
