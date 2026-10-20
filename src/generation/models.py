"""Data models and schemas for LLM answer generation and citations."""

from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from src.retrieval.models import RetrievedChunk


class AnswerStatus(str, Enum):
    """Categorical classification of the generated answer's grounding state."""
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    ABSTAINED_OUT_OF_SCOPE = "abstained_out_of_scope"
    ABSTAINED_INSUFFICIENT_EVIDENCE = "abstained_insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    ERROR = "error"


class Citation(BaseModel):
    """A verified citation linked directly to a retrieved source chunk in the corpus."""
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(description="Source document identifier (e.g. 'P1')")
    document_title: str = Field(description="Full title of the source document")
    physical_page_number: int = Field(description="One-based physical page number in the PDF")
    printed_page_number: Optional[int] = Field(
        default=None,
        description="Printed page number if verified; None for preprints"
    )
    chunk_id: str = Field(description="Deterministic chunk ID (e.g. 'P1_p03_c04')")
    excerpt: str = Field(description="Short verbatim excerpt from the chunk supporting the claim")

    def formatted_citation(self) -> str:
        """Render a readable citation string for display."""
        if self.printed_page_number is not None:
            page_desc = f"p. {self.physical_page_number} (printed p. {self.printed_page_number})"
        else:
            page_desc = f"p. {self.physical_page_number}"
        return f"[{self.document_id}, {page_desc}, Chunk {self.chunk_id}]"


class GenerationConfig(BaseModel):
    """Configuration options for the answer generation layer."""
    model_config = ConfigDict(extra="ignore")

    provider_type: str = Field(
        default="auto",
        description="LLM provider: 'auto', 'mock', 'openai', 'gemini', 'groq', 'ollama'"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model identifier (e.g. 'gpt-4o-mini', 'gemini-1.5-flash')"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; 0.0 for deterministic factual generation"
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum generation length in tokens"
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=65536,
        description="Provider-specific completion cap when supported; otherwise max_tokens is used"
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="HTTP request timeout in seconds"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context"
    )
    min_score: Optional[float] = Field(
        default=None,
        description="Optional minimum similarity score threshold"
    )
    retrieval_mode: str = Field(
        default="dense",
        description="Retrieval subsystem mode: 'dense' or 'hybrid'"
    )


class GeneratedAnswer(BaseModel):
    """Structured response containing the grounded answer, citations, and provenance."""
    model_config = ConfigDict(extra="ignore")

    question: str = Field(description="The original user research question")
    answer: str = Field(description="Synthesized evidence-grounded answer or abstention message")
    status: AnswerStatus = Field(description="Categorical grounding status of the answer")
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of verified citations linking claims to retrieved chunks"
    )
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="The actual retrieved chunks supplied to the LLM context"
    )
    confidence_notes: Optional[str] = Field(
        default=None,
        description="Explanation of missing information, partial evidence, or conflicts"
    )
    model_name: str = Field(description="Model identifier used for generation")
    provider_name: str = Field(description="Provider name used (e.g. 'openai', 'gemini', 'mock')")
    retrieval_mode: Optional[str] = Field(
        default=None,
        description="Retrieval mode used for context: 'dense' or 'hybrid'"
    )
    latency_ms: float = Field(default=0.0, description="Total generation latency in milliseconds")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in completion")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed")

