"""Targeted UI tests for the Streamlit Research Intelligence Engine using AppTest.

Verifies:
1. Startup and clean rendering without unhandled exceptions.
2. All 3 tabs (Grounded Q&A, Paper Ingestion, System Status) render correctly.
3. Sidebar controls: provider selection, retrieval mode, top-k slider, model override.
4. Sample question button state updates.
5. Evidence-grounded answer rendering, citations cards, and retrieved context inspector.
6. Abstention rendering on out-of-scope / insufficient-evidence queries.
7. Paper ingestion flow (corpus selection and dry-run indexing).
8. Error handling and credentials protection (no secrets exposed).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure utf-8 terminal output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from streamlit.testing.v1 import AppTest

from src.generation.models import GeneratedAnswer, AnswerStatus, Citation
from src.retrieval.models import RetrievedChunk
from src.ui.service import IngestionReport


@pytest.fixture
def app_path():
    """Return path to Streamlit application."""
    return str(Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py")


def test_ui_initial_render_and_sections(app_path):
    """Verify application starts and all three primary tabs render without error."""
    at = AppTest.from_file(app_path, default_timeout=25)
    at.run()

    # Verify no unhandled exceptions
    assert len(at.exception) == 0, f"AppTest raised unexpected exception: {[e.message for e in at.exception]}"

    # Verify tabs exist
    assert len(at.tabs) == 3

    # Verify sidebar content
    sidebar_texts = " ".join([m.value for m in at.sidebar.markdown])
    assert "Research Engine" in sidebar_texts
    assert "Provider & Model Settings" in sidebar_texts
    assert "Retrieval Settings" in sidebar_texts
    assert "Indexed Corpus" in sidebar_texts


def test_sidebar_controls_behavior(app_path):
    """Verify provider selector, retrieval mode radio, top-k slider, and model override."""
    at = AppTest.from_file(app_path, default_timeout=25)
    at.run()
    assert len(at.exception) == 0

    # 1. Provider Selectbox
    provider_sb = at.sidebar.selectbox[0]
    assert "mock" in provider_sb.options
    assert "gemini" in provider_sb.options
    assert "openai" in provider_sb.options
    provider_sb.select("mock")
    at.run()
    assert at.sidebar.selectbox[0].value == "mock"

    # 2. Retrieval Mode Radio
    mode_radio = at.sidebar.radio[0]
    assert "Dense Vector Only" in mode_radio.options
    mode_radio.set_value("Dense Vector Only")
    at.run()
    assert at.sidebar.radio[0].value == "Dense Vector Only"

    # 3. Top-k Slider
    slider = at.sidebar.slider[0]
    assert slider.min == 1
    assert slider.max == 10
    slider.set_value(8)
    at.run()
    assert at.sidebar.slider[0].value == 8

    # 4. Model Identifier Override
    model_input = at.sidebar.text_input[0]
    model_input.input("gemini-test-override")
    at.run()
    assert at.sidebar.text_input[0].value == "gemini-test-override"


def test_sample_question_buttons_sync(app_path):
    """Verify that clicking sample question buttons accurately updates the question text area."""
    at = AppTest.from_file(app_path, default_timeout=25)
    at.run()
    assert len(at.exception) == 0

    # 1. Click Self-RAG button
    [b for b in at.button if "Self-RAG" in b.label][0].click()
    at.run()
    assert "Self-RAG decide when to retrieve" in at.text_area[0].value

    # 2. Click DPR button
    [b for b in at.button if "DPR" in b.label][0].click()
    at.run()
    assert "BM25 outperform DPR on SQuAD" in at.text_area[0].value

    # 3. Click Out-of-Scope
    [b for b in at.button if "Out-of-Scope" in b.label][0].click()
    at.run()
    assert "2024 ICC Men's T20 World Cup" in at.text_area[0].value

    # 4. Click RAG button
    [b for b in at.button if "RAG:" in b.label][0].click()
    at.run()
    assert "RAG-Sequence and RAG-Token differ" in at.text_area[0].value


def test_grounded_qa_rendering_with_mock(app_path):
    """Verify grounded answer, citations, page provenance, and context passage inspector display."""
    mock_answer = GeneratedAnswer(
        question="How do RAG-Sequence and RAG-Token differ in generating sequences?",
        answer="RAG-Sequence marginalizes over sequences; RAG-Token marginalizes per token.",
        status=AnswerStatus.ANSWERED,
        citations=[
            Citation(
                chunk_id="P1_p04_c01",
                document_id="P1",
                document_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                physical_page_number=4,
                printed_page_number=4,
                excerpt="minimize the negative marginal log-likelihood of each target",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="P1_p04_c01",
                document_id="P1",
                document_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                physical_page_index=3,
                physical_page_number=4,
                printed_page_number=4,
                chunk_index_in_page=0,
                chunk_index_in_doc=12,
                start_char_idx=0,
                end_char_idx=200,
                char_count=200,
                word_count=35,
                text="minimize the negative marginal log-likelihood of each target...",
                score=0.8842,
                retrieval_method="hybrid",
            )
        ],
        model_name="gemini-flash-lite-latest",
        provider_name="gemini",
        retrieval_mode="hybrid",
        latency_ms=145.2,
        prompt_tokens=220,
        completion_tokens=65,
        total_tokens=285,
        confidence_notes="Evidence supported by verified chunk P1_p04_c01.",
    )

    with patch("src.ui.service.ResearchEngineService.ask_question", return_value=mock_answer):
        at = AppTest.from_file(app_path, default_timeout=25)
        at.run()
        assert len(at.exception) == 0

        # Submit question form
        submit_btn = [b for b in at.button if b.label == "Generate Answer"][0]
        submit_btn.click()
        at.run()
        assert len(at.exception) == 0

        # Check rendered contents
        markdown_str = " ".join([m.value for m in at.markdown])
        assert "Answered Grounded" in markdown_str
        assert "RAG-Sequence marginalizes over sequences" in markdown_str
        assert "Verified Citations (1)" in markdown_str
        assert "Physical Page: <b>4</b>" in markdown_str
        assert "Printed p. 4" in markdown_str
        assert "P1_p04_c01" in markdown_str
        assert "minimize the negative marginal log-likelihood" in markdown_str

        # Check confidence note
        info_str = " ".join([inf.value for inf in at.info])
        assert "Evidence supported by verified chunk" in info_str


def test_abstention_rendering_with_mock(app_path):
    """Verify that out-of-scope / insufficient-evidence responses display clear abstentions without citations."""
    mock_abstention = GeneratedAnswer(
        question="Who won the 2024 ICC Men's T20 World Cup?",
        answer="The provided research documents do not contain information about the 2024 ICC Men's T20 World Cup.",
        status=AnswerStatus.ABSTAINED_OUT_OF_SCOPE,
        citations=[],
        retrieved_chunks=[],
        model_name="gemini-flash-lite-latest",
        provider_name="gemini",
        retrieval_mode="hybrid",
        latency_ms=85.0,
        confidence_notes="Out-of-scope query.",
    )

    with patch("src.ui.service.ResearchEngineService.ask_question", return_value=mock_abstention):
        at = AppTest.from_file(app_path, default_timeout=25)
        at.run()
        assert len(at.exception) == 0

        submit_btn = [b for b in at.button if b.label == "Generate Answer"][0]
        submit_btn.click()
        at.run()
        assert len(at.exception) == 0

        markdown_str = " ".join([m.value for m in at.markdown])
        assert "Abstained: Out of Scope" in markdown_str
        assert "The provided research documents do not contain information" in markdown_str
        assert "Verified Citations (0)" in markdown_str


def test_paper_ingestion_ui_flow(app_path):
    """Verify paper ingestion triggers successfully and displays created chunk metrics."""
    mock_report = IngestionReport(
        success=True,
        document_id="P5",
        title="Ragas: Automated Evaluation of Retrieval Augmented Generation",
        filename="P5_RAGAS_Es_2023.pdf",
        pages_extracted=8,
        chunks_created=39,
        chunks_indexed=39,
        total_collection_points=396,
        sample_chunks=[
            {
                "chunk_id": "P5_p01_c01",
                "page": 1,
                "words": 42,
                "preview": "Ragas: Automated Evaluation of Retrieval Augmented Generation...",
            }
        ],
    )

    with patch("src.ui.service.ResearchEngineService.ingest_paper", return_value=mock_report):
        at = AppTest.from_file(app_path, default_timeout=25)
        at.run()
        assert len(at.exception) == 0

        # Click dry run checkbox to true
        at.checkbox[0].check()
        at.run()

        # Click ingest button
        ingest_btn = [b for b in at.button if "Ingest & Index Document" in b.label][0]
        ingest_btn.click()
        at.run()
        assert len(at.exception) == 0

        # Check success banner
        success_str = " ".join([s.value for s in at.success])
        assert "Successfully ingested and indexed" in success_str
        assert "Ragas: Automated Evaluation" in success_str
        assert "P5" in success_str


def test_error_handling_sanitization_in_ui(app_path):
    """Verify error responses display sanitized failure badge without exposing secrets or tracebacks."""
    mock_error_ans = GeneratedAnswer(
        question="Trigger error query",
        answer="Authentication failed for provider 'gemini'. Please ensure the required API key is set in your environment or .env file.",
        status=AnswerStatus.ERROR,
        citations=[],
        retrieved_chunks=[],
        model_name="gemini-flash-lite-latest",
        provider_name="gemini",
        retrieval_mode="hybrid",
        latency_ms=10.0,
        confidence_notes="Credentials were not exposed.",
    )

    with patch("src.ui.service.ResearchEngineService.ask_question", return_value=mock_error_ans):
        at = AppTest.from_file(app_path, default_timeout=25)
        at.run()
        assert len(at.exception) == 0

        submit_btn = [b for b in at.button if b.label == "Generate Answer"][0]
        submit_btn.click()
        at.run()
        assert len(at.exception) == 0

        markdown_str = " ".join([m.value for m in at.markdown])
        assert "Generation Error" in markdown_str
        assert "Authentication failed for provider" in markdown_str
        # Verify no secret keywords leaked
        assert "sk-" not in markdown_str
        assert "AIzaSy" not in markdown_str
        assert "Traceback (most recent call last)" not in markdown_str


def test_full_user_journey_with_mock_provider(app_path):
    """Verify complete un-mocked user journey through AppTest using the offline mock provider."""
    at = AppTest.from_file(app_path, default_timeout=90)
    at.run(timeout=90)
    assert len(at.exception) == 0

    # 1. Verify default or set provider to 'mock'
    provider_sb = at.sidebar.selectbox[0]
    provider_sb.select("mock")
    at.run(timeout=90)
    assert len(at.exception) == 0

    # 2. Click the RAG sample question button
    rag_btn = [b for b in at.button if "RAG:" in b.label][0]
    rag_btn.click()
    at.run(timeout=90)
    assert len(at.exception) == 0
    assert "RAG-Sequence and RAG-Token differ" in at.text_area[0].value

    # 3. Submit question to generate grounded answer (PyTorch model initializes here)
    submit_btn = [b for b in at.button if b.label == "Generate Answer"][0]
    submit_btn.click()
    at.run(timeout=90)
    assert len(at.exception) == 0

    # Verify grounded answer is rendered
    markdown_str = " ".join([m.value for m in at.markdown])
    assert "Answered Grounded" in markdown_str
    assert "In RAG-Sequence, the model uses the same retrieved document" in markdown_str
    assert "Verified Citations" in markdown_str
    assert "P1_p04_c01" in markdown_str
    assert "Physical Page:" in markdown_str

    # 4. Now click out-of-scope question to verify abstention journey
    oos_btn = [b for b in at.button if "Out-of-Scope" in b.label][0]
    oos_btn.click()
    at.run(timeout=90)
    assert len(at.exception) == 0
    assert "2024 ICC Men's T20 World Cup" in at.text_area[0].value

    submit_btn2 = [b for b in at.button if b.label == "Generate Answer"][0]
    submit_btn2.click()
    at.run(timeout=90)
    assert len(at.exception) == 0

    # Verify abstention rendered
    markdown_str2 = " ".join([m.value for m in at.markdown])
    assert "Abstained:" in markdown_str2
    assert "Verified Citations (0)" in markdown_str2

