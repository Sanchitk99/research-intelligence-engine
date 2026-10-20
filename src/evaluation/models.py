"""Data models and schemas for retrieval evaluation."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class RetrievedChunkEval(BaseModel):
    """Evaluation record for an individual retrieved chunk."""
    model_config = ConfigDict(extra="ignore")

    rank: int = Field(description="1-based retrieval rank")
    chunk_id: str = Field(description="Chunk identifier (e.g. 'P1_p01_c01')")
    document_id: str = Field(description="Source document identifier (e.g. 'P1')")
    physical_page_number: int = Field(description="Physical page number in PDF")
    printed_page_number: Optional[int] = Field(default=None, description="Printed page number if verified")
    score: float = Field(description="Cosine similarity score")
    is_relevant: bool = Field(description="Whether this chunk matches ground-truth target")
    snippet: str = Field(description="Short preview of chunk text")


class QuestionEvaluationResult(BaseModel):
    """Detailed evaluation result for a single question."""
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(description="Question ID (e.g. 'Q01')")
    question_text: str = Field(description="Question text")
    question_type: str = Field(description="Category (Factual, Multi-paper, Comparison, Limitations)")
    expected_documents: List[str] = Field(description="List of expected document IDs")
    expected_pages: Dict[str, List[int]] = Field(description="Expected physical page numbers per document")
    supporting_excerpt: str = Field(default="", description="Ground-truth excerpt snippet from annotations")
    retrieved_chunks: List[RetrievedChunkEval] = Field(
        default_factory=list,
        description="Retrieved chunks ordered by rank"
    )
    first_relevant_rank: Optional[int] = Field(
        default=None,
        description="1-based rank of first relevant chunk, or None if not found in top-k"
    )
    reciprocal_rank: float = Field(default=0.0, description="1 / first_relevant_rank or 0.0")
    hit_at_1: bool = Field(default=False, description="Whether a relevant chunk is at rank 1")
    hit_at_3: bool = Field(default=False, description="Whether a relevant chunk is in top 3")
    hit_at_5: bool = Field(default=False, description="Whether a relevant chunk is in top 5")
    documents_covered_at_5: List[str] = Field(
        default_factory=list,
        description="Expected document IDs that had at least one relevant chunk in top 5"
    )
    document_coverage_at_5: float = Field(
        default=0.0,
        description="Fraction of expected documents with relevant evidence in top 5"
    )
    all_documents_covered_at_5: bool = Field(
        default=False,
        description="Whether ALL expected documents were retrieved in top 5"
    )
    any_document_covered_at_5: bool = Field(
        default=False,
        description="Whether AT LEAST ONE expected document was retrieved in top 5"
    )
    target_pages_total: int = Field(
        default=0,
        description="Total count of annotated target pages across expected documents"
    )
    target_pages_covered_at_5: int = Field(
        default=0,
        description="Count of distinct annotated target pages retrieved in top 5"
    )
    target_page_recall_at_5: float = Field(
        default=0.0,
        description="Fraction of annotated target pages retrieved in top 5"
    )


class EvaluationSummary(BaseModel):
    """Aggregate metrics summary across the evaluation corpus."""
    model_config = ConfigDict(extra="ignore")

    total_questions: int = Field(description="Total questions in evaluation dataset")
    evaluated_questions: int = Field(description="Number of questions evaluated")
    excluded_questions: int = Field(description="Number of questions excluded (e.g. out-of-scope)")
    recall_at_1: float = Field(description="Fraction of questions with relevant evidence at rank 1 (Hit@1)")
    recall_at_3: float = Field(description="Fraction of questions with relevant evidence in top 3 (Hit@3)")
    recall_at_5: float = Field(description="Fraction of questions with relevant evidence in top 5 (Hit@5)")
    hit_at_1: float = Field(default=0.0, description="Alias for recall_at_1 (binary Hit@1)")
    hit_at_3: float = Field(default=0.0, description="Alias for recall_at_3 (binary Hit@3)")
    hit_at_5: float = Field(default=0.0, description="Alias for recall_at_5 (binary Hit@5)")
    mrr: float = Field(description="Mean Reciprocal Rank across evaluated questions (truncated at top_k=5, i.e. MRR@5)")
    mrr_at_5: float = Field(default=0.0, description="Explicit Mean Reciprocal Rank truncated at rank cutoff 5 (MRR@5)")
    reciprocal_rank_sum: float = Field(default=0.0, description="Sum of reciprocal ranks across evaluated questions")
    mean_document_coverage_at_5: float = Field(
        description="Average fraction of required documents retrieved in top 5 across all questions"
    )
    mean_target_page_recall_at_5: float = Field(
        default=0.0,
        description="Average fraction of annotated target pages retrieved in top 5"
    )
    multi_doc_questions_count: int = Field(
        default=0,
        description="Number of multi-document questions evaluated"
    )
    multi_doc_any_coverage_at_5: float = Field(
        default=0.0,
        description="Fraction of multi-doc questions where at least one expected document was retrieved in top 5"
    )
    multi_doc_full_coverage_at_5: float = Field(
        default=0.0,
        description="Fraction of multi-doc questions where ALL expected documents were retrieved in top 5"
    )
    multi_doc_mean_coverage_at_5: float = Field(
        default=0.0,
        description="Mean document coverage strictly across multi-document questions"
    )
    metrics_by_type: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Metric breakdown by question type (Factual, Comparison, Multi-paper, Limitations)"
    )
    questions_missing_in_top_5: List[str] = Field(
        default_factory=list,
        description="List of question IDs where no relevant evidence appeared in top 5"
    )
    execution_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of evaluation execution"
    )

