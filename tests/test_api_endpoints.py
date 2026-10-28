"""FastAPI endpoint integration and contract tests for the Research Intelligence Engine."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_research_service
from src.generation.models import GeneratedAnswer, AnswerStatus, Citation
from src.retrieval.models import RetrievedChunk
from src.ui.service import PaperSummary, IngestionReport


@pytest.fixture
def mock_service():
    """Create a mock ResearchEngineService for fast endpoint testing without loading weights."""
    mock = MagicMock()

    # Mock get_system_status
    mock.get_system_status.return_value = {
        "collection_name": "test_collection",
        "total_points": 396,
        "indexed_papers_count": 5,
        "papers": [
            PaperSummary(document_id="P1", title="RAG Paper", total_pages=10, total_chunks=80),
            PaperSummary(document_id="P2", title="DPR Paper", total_pages=12, total_chunks=90),
        ],
        "providers": {
            "mock": {"configured": True, "default_model": "mock-evaluator"},
            "gemini": {"configured": True, "default_model": "gemini-flash-lite-latest"},
        },
    }

    # Mock get_indexed_papers
    mock.get_indexed_papers.return_value = [
        PaperSummary(document_id="P1", title="RAG Paper", total_pages=10, total_chunks=80),
        PaperSummary(document_id="P2", title="DPR Paper", total_pages=12, total_chunks=90),
    ]

    # Mock ask_question (Grounded Answer)
    mock.ask_question.return_value = GeneratedAnswer(
        question="What generator model does RAG use?",
        answer="RAG uses BART-large as its generator.",
        status=AnswerStatus.ANSWERED,
        citations=[
            Citation(
                chunk_id="P1_p04_c01",
                document_id="P1",
                document_title="RAG Paper",
                physical_page_number=4,
                printed_page_number=4,
                excerpt="only fine-tuning the query encoder BERTq and the BART generator.",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="P1_p04_c01",
                document_id="P1",
                document_title="RAG Paper",
                physical_page_index=3,
                physical_page_number=4,
                chunk_index_in_page=0,
                chunk_index_in_doc=15,
                start_char_idx=0,
                end_char_idx=100,
                char_count=100,
                word_count=18,
                text="only fine-tuning the query encoder BERTq and the BART generator.",
                score=0.85,
            )
        ],
        model_name="mock-model",
        provider_name="mock",
        retrieval_mode="hybrid",
        latency_ms=125.4,
        confidence_notes="Supported by P1",
        prompt_tokens=500,
        completion_tokens=25,
        total_tokens=525,
    )

    # Mock ingest_paper
    mock.ingest_paper.return_value = IngestionReport(
        success=True,
        document_id="P5",
        title="Ragas Evaluation",
        filename="P5_RAGAS_Es_2023.pdf",
        pages_extracted=8,
        chunks_created=39,
        chunks_indexed=39,
        total_collection_points=396,
        sample_chunks=[{"chunk_id": "P5_p01_c01", "preview": "Ragas: Automated Evaluation"}],
    )

    return mock


@pytest.fixture
def client(mock_service):
    """FastAPI TestClient with overridden service dependency."""
    app.dependency_overrides[get_research_service] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    """Verify /api/system/health returns 200 and expected metadata."""
    res = client.get("/api/system/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert data["version"] == "1.0.0"


def test_system_status_endpoint(client, mock_service):
    """Verify /api/system/status serializes collection metrics and sanitized provider statuses."""
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["collection_name"] == "test_collection"
    assert data["total_points"] == 396
    assert data["indexed_papers_count"] == 5
    assert len(data["papers"]) == 2
    assert "mock" in data["providers"]
    assert data["providers"]["mock"]["configured"] is True
    # Ensure zero secrets leaked
    assert "AIzaSy" not in str(data)
    assert "sk-" not in str(data)


def test_research_ask_grounded(client, mock_service):
    """Verify /api/research/ask generates grounded answers with verified citations."""
    payload = {
        "question": "What generator model does RAG use?",
        "provider": "mock",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    }
    res = client.post("/api/research/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["question"] == payload["question"]
    assert "BART-large" in data["answer"]
    assert data["status"] == "answered"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "P1_p04_c01"
    assert data["citations"][0]["physical_page_number"] == 4
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["score"] == 0.85


def test_research_ask_empty_question_fails(client):
    """Verify /api/research/ask rejects empty or whitespace-only questions with 400 Bad Request."""
    res = client.post("/api/research/ask", json={"question": "   \n\t  "})
    assert res.status_code == 400
    assert "empty or whitespace-only" in res.json()["detail"]


def test_research_ask_abstention(client, mock_service):
    """Verify /api/research/ask properly serializes abstention responses with zero citations."""
    mock_service.ask_question.return_value = GeneratedAnswer(
        question="Who won the 2024 ICC Men's T20 World Cup?",
        answer="The provided research evidence does not contain information to address this question.",
        status=AnswerStatus.ABSTAINED_OUT_OF_SCOPE,
        citations=[],
        retrieved_chunks=[],
        model_name="mock-model",
        provider_name="mock",
        retrieval_mode="hybrid",
        latency_ms=45.2,
        confidence_notes="Out of scope query.",
    )

    res = client.post("/api/research/ask", json={"question": "Who won the 2024 ICC Men's T20 World Cup?"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "abstained_out_of_scope"
    assert len(data["citations"]) == 0
    assert "does not contain information" in data["answer"]


def test_get_papers_endpoint(client, mock_service):
    """Verify /api/papers lists indexed papers."""
    res = client.get("/api/papers")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 2
    assert data[0]["document_id"] == "P1"
    assert data[1]["document_id"] == "P2"


def test_get_corpus_pdfs_endpoint(client):
    """Verify /api/papers/corpus lists actual benchmark PDFs from evaluation/corpus/papers/."""
    res = client.get("/api/papers/corpus")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 5
    filenames = [p["filename"] for p in data]
    assert any("P1_RAG" in f for f in filenames)
    assert any("P5_RAGAS" in f for f in filenames)


def test_ingest_paper_dry_run(client, mock_service):
    """Verify /api/papers/ingest handles dry-run ingestion correctly."""
    payload = {
        "filename": "P5_RAGAS_Es_2023.pdf",
        "dry_run": True,
    }
    res = client.post("/api/papers/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["document_id"] == "P5"
    assert data["chunks_created"] == 39
    mock_service.ingest_paper.assert_called_once()


def test_ingest_nonexistent_paper_404(client):
    """Verify /api/papers/ingest returns 404 for missing PDF file."""
    payload = {"filename": "non_existent_paper_12345.pdf"}
    res = client.post("/api/papers/ingest", json=payload)
    assert res.status_code == 404


def test_upload_non_pdf_fails(client):
    """Verify /api/papers/upload rejects non-PDF file extensions."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    res = client.post("/api/papers/upload", files=files)
    assert res.status_code == 400
    assert "must have a .pdf extension" in res.json()["detail"]


def test_upload_pdf_dry_run(client, mock_service):
    """Verify /api/papers/upload correctly processes dry-run mode and reports 0 chunks indexed."""
    mock_service.ingest_paper.return_value = IngestionReport(
        success=True,
        document_id="P6",
        title="Test Upload Paper",
        filename="test_upload.pdf",
        pages_extracted=5,
        chunks_created=20,
        chunks_indexed=0,
        total_collection_points=396,
        dry_run=True,
    )
    files = {"file": ("test_upload.pdf", b"%PDF-1.4 synthetic content", "application/pdf")}
    res = client.post("/api/papers/upload", files=files, data={"dry_run": "true"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["dry_run"] is True
    assert data["chunks_indexed"] == 0
    assert data["chunks_created"] == 20


def test_upload_pdf_actual_ingestion(client, mock_service):
    """Verify /api/papers/upload correctly indexes chunks and reports positive indexed count."""
    mock_service.ingest_paper.return_value = IngestionReport(
        success=True,
        document_id="P6",
        title="Test Upload Paper",
        filename="test_upload.pdf",
        pages_extracted=5,
        chunks_created=20,
        chunks_indexed=20,
        total_collection_points=416,
        dry_run=False,
    )
    files = {"file": ("test_upload.pdf", b"%PDF-1.4 synthetic content", "application/pdf")}
    res = client.post("/api/papers/upload", files=files, data={"dry_run": "false"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["dry_run"] is False
    assert data["chunks_indexed"] == 20
    assert data["total_collection_points"] == 416
