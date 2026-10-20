"""Targeted integration tests verifying the demonstrable end-to-end user workflow.

Tests:
1. Engine service status and indexed corpus metadata.
2. PDF ingestion, page extraction, and text chunking pipeline.
3. Evidence-grounded Q&A with citation verification.
4. Abstention handling on out-of-scope queries.
5. Error handling and input sanitization.
"""

import sys
from pathlib import Path
import pytest

# Ensure utf-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from src.ui.service import ResearchEngineService, IngestionReport, PaperSummary
from src.generation.models import AnswerStatus, GeneratedAnswer


@pytest.fixture(scope="module")
def engine_service():
    """Shared service fixture for demonstration workflow tests."""
    service = ResearchEngineService(
        qdrant_path="data/qdrant",
        collection_name="research_corpus",
        chunks_file="data/chunked/corpus_chunks.json",
    )
    yield service
    service.close()


def test_service_initialization_and_status(engine_service):
    """Verify service correctly inspects Qdrant collection and indexed papers."""
    status = engine_service.get_system_status()
    assert status["collection_name"] == "research_corpus"
    assert status["total_points"] >= 396
    assert status["indexed_papers_count"] >= 5
    assert len(status["papers"]) >= 5

    # Check paper structure
    paper_ids = [p.document_id for p in status["papers"]]
    assert "P1" in paper_ids
    assert "P5" in paper_ids


def test_ingestion_and_chunking_workflow(engine_service):
    """Verify paper ingestion pipeline on actual PDF (dry_run mode for test isolation)."""
    sample_pdf = Path("evaluation/corpus/papers/P5_RAGAS_Es_2023.pdf")
    assert sample_pdf.is_file(), f"Sample paper not found: {sample_pdf}"

    report: IngestionReport = engine_service.ingest_paper(
        pdf_path=sample_pdf,
        doc_id="P5",
        doc_title="Ragas: Automated Evaluation of Retrieval Augmented Generation",
        dry_run=True,
    )

    assert report.success is True
    assert report.document_id == "P5"
    assert report.pages_extracted == 8
    assert report.chunks_created > 0
    assert len(report.sample_chunks) > 0
    assert "chunk_id" in report.sample_chunks[0]
    assert report.sample_chunks[0]["page"] >= 1


def test_grounded_qa_with_mock_provider(engine_service):
    """Verify end-to-end question answering and citation verification using mock provider."""
    # Question with supported evidence in top chunks
    question = "How do RAG-Sequence and RAG-Token differ in generating sequences?"
    ans: GeneratedAnswer = engine_service.ask_question(
        question=question,
        provider_name="mock",
        retrieval_mode="hybrid",
        top_k=5,
    )

    assert ans.question == question
    assert ans.status in (AnswerStatus.ANSWERED, AnswerStatus.PARTIALLY_ANSWERED)
    assert len(ans.retrieved_chunks) > 0
    assert len(ans.citations) > 0

    # Verify citation provenance
    first_citation = ans.citations[0]
    assert first_citation.document_id == "P1"
    assert first_citation.physical_page_number in (3, 4, 5)
    assert len(first_citation.excerpt.strip()) > 0
    assert first_citation.chunk_id.startswith("P1_")


def test_out_of_scope_abstention_handling(engine_service):
    """Verify that unanswerable/out-of-scope queries produce clear abstentions without citations."""
    out_of_scope_q = "Who won the 2024 ICC Men's T20 World Cup?"
    ans: GeneratedAnswer = engine_service.ask_question(
        question=out_of_scope_q,
        provider_name="mock",
        retrieval_mode="hybrid",
        top_k=5,
    )

    # Must abstain with insufficient evidence or out of scope
    assert ans.status in (
        AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE,
        AnswerStatus.ABSTAINED_OUT_OF_SCOPE,
    )
    # Must NOT fabricate or attach citations
    assert len(ans.citations) == 0


def test_empty_query_error_handling(engine_service):
    """Verify empty or whitespace questions are rejected cleanly with safe status."""
    ans: GeneratedAnswer = engine_service.ask_question(
        question="   ",
        provider_name="mock",
    )
    assert ans.status == AnswerStatus.ERROR
    assert "Question cannot be empty" in ans.answer
    assert len(ans.citations) == 0


def test_missing_pdf_ingestion_error(engine_service):
    """Verify ingesting a non-existent PDF returns a clean failure report without throwing an exception."""
    report: IngestionReport = engine_service.ingest_paper(
        pdf_path="non_existent_paper.pdf",
    )
    assert report.success is False
    assert "does not exist" in report.error_message.lower()
