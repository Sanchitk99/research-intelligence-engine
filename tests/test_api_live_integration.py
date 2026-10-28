"""Live end-to-end integration test for the FastAPI backend using real RAG engine components and mock provider."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.generation.models import AnswerStatus


@pytest.fixture(scope="module")
def live_client():
    """Create live client, connecting to running server on port 8000 or falling back to in-process TestClient."""
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:8000/api/system/health", timeout=1.5)
        if r.status_code == 200:
            with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0) as client:
                yield client
            return
    except Exception:
        pass

    # Fallback to TestClient if standalone server is not running
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        yield client


def test_live_health_check(live_client):
    """Verify live API health check endpoint."""
    res = live_client.get("/api/system/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "Research Intelligence Engine API"


def test_live_system_status(live_client):
    """Verify live system status queries the actual Qdrant collection and corpus chunks."""
    res = live_client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["collection_name"] == "research_corpus"
    assert data["total_points"] >= 396
    assert data["indexed_papers_count"] >= 5
    assert len(data["papers"]) >= 5
    assert "mock" in data["providers"]
    assert data["providers"]["mock"]["configured"] is True


def test_live_get_indexed_papers(live_client):
    """Verify live /api/papers endpoint returns all 5 canonical research papers."""
    res = live_client.get("/api/papers")
    assert res.status_code == 200
    papers = res.json()
    doc_ids = [p["document_id"] for p in papers]
    assert "P1" in doc_ids
    assert "P2" in doc_ids
    assert "P3" in doc_ids
    assert "P4" in doc_ids
    assert "P5" in doc_ids


def test_live_get_corpus_pdfs(live_client):
    """Verify live /api/papers/corpus finds all benchmark PDF files."""
    res = live_client.get("/api/papers/corpus")
    assert res.status_code == 200
    corpus = res.json()
    assert len(corpus) >= 5
    filenames = [c["filename"] for c in corpus]
    assert any("P1_RAG" in f for f in filenames)
    assert any("P5_RAGAS" in f for f in filenames)


def test_live_grounded_qa_with_mock_provider(live_client):
    """Verify live end-to-end question answering with Hybrid RRF and verified page citations."""
    payload = {
        "question": "How do RAG-Sequence and RAG-Token differ in generating sequences?",
        "provider": "mock",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    }
    res = live_client.post("/api/research/ask", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "answered"
    assert "In RAG-Sequence, the model uses the same retrieved document" in data["answer"]
    assert len(data["citations"]) >= 1

    citation = data["citations"][0]
    assert citation["document_id"] == "P1"
    assert citation["physical_page_number"] in (3, 4)
    assert len(citation["excerpt"]) > 10
    assert len(data["retrieved_chunks"]) == 5
    assert data["latency_ms"] > 0


def test_live_out_of_scope_abstention_with_mock_provider(live_client):
    """Verify live out-of-scope query produces explicit abstention and strictly 0 citations."""
    payload = {
        "question": "Who won the 2024 ICC Men's T20 World Cup?",
        "provider": "mock",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    }
    res = live_client.post("/api/research/ask", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "abstained" in data["status"]
    assert len(data["citations"]) == 0
    assert "does not contain" in data["answer"].lower()


def test_live_paper_dry_run_ingestion(live_client):
    """Verify live dry-run ingestion processes PDF without modifying Qdrant vector counts."""
    payload = {
        "filename": "P5_RAGAS_Es_2023.pdf",
        "dry_run": True,
    }
    res = live_client.post("/api/papers/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["document_id"] == "P5"
    assert data["pages_extracted"] == 8
    assert data["chunks_created"] == 39
    assert data["chunks_indexed"] == 0
    assert data["dry_run"] is True
    assert len(data["sample_chunks"]) > 0
