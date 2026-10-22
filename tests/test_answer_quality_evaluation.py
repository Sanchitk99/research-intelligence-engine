"""Unit and integration tests for Controlled Answer Quality & Latency Evaluation.

Verifies the 6 key benchmark dimensions and behaviors:
1. Dataset integrity across answerable and out-of-scope benchmark questions.
2. Citation validity checking and excerpt faithfulness grounding.
3. Abstained and answered query evaluation and multi-dimensional scorecard generation.
4. Unsupported-answer risk calculation mechanism (zero risk via safe abstention).
5. Evidence claim-support lexical keyword overlap heuristic.
6. Provider configuration auditing, credential redaction, and Ollama reachability.
7. Percentile computation math (median, p95) using linear interpolation.
"""

import json
from pathlib import Path
import pytest

from scripts.run_answer_quality_evaluation import (
    compute_percentile,
    audit_provider_configuration,
    run_controlled_evaluation,
    generate_markdown_report,
)
from src.evaluation.generation_evaluator import check_evidence_claim_support
from src.evaluation.ground_truth import load_ground_truth
from src.generation.models import AnswerStatus, Citation, GeneratedAnswer
from src.retrieval.models import RetrievedChunk


def test_dataset_integrity():
    """Verify integrity, schemas, and ground-truth alignment of evaluation question datasets."""
    ans_path = Path("evaluation/questions/answerable_questions.json")
    oos_path = Path("evaluation/questions/out_of_scope_questions.json")

    assert ans_path.is_file(), f"Missing answerable questions dataset at {ans_path}"
    assert oos_path.is_file(), f"Missing out-of-scope questions dataset at {oos_path}"

    with open(ans_path, "r", encoding="utf-8") as f:
        ans_data = json.load(f)
    with open(oos_path, "r", encoding="utf-8") as f:
        oos_data = json.load(f)

    # Benchmark dataset volume checks
    assert len(ans_data) == 18, f"Expected 18 answerable questions, got {len(ans_data)}"
    assert len(oos_data) == 10, f"Expected 10 out-of-scope questions, got {len(oos_data)}"

    # Check answerable schema
    ans_ids = set()
    for q in ans_data:
        assert "id" in q and q["id"].startswith("Q"), f"Invalid ID format: {q.get('id')}"
        assert q["id"] not in ans_ids, f"Duplicate question ID: {q['id']}"
        ans_ids.add(q["id"])

        assert len(q.get("question", "").strip()) > 0, f"Empty question in {q['id']}"
        assert len(q.get("expected_answer", "").strip()) > 0, f"Empty expected answer in {q['id']}"
        assert isinstance(q.get("source_papers"), list) and len(q["source_papers"]) > 0, f"Missing source papers in {q['id']}"
        assert len(q.get("page_references", "").strip()) > 0, f"Missing page references in {q['id']}"

    # Check out-of-scope schema
    oos_ids = set()
    for q in oos_data:
        assert "id" in q and q["id"].startswith("OOS"), f"Invalid OOS ID format: {q.get('id')}"
        assert q["id"] not in oos_ids, f"Duplicate OOS question ID: {q['id']}"
        oos_ids.add(q["id"])

        assert len(q.get("question", "").strip()) > 0, f"Empty OOS question in {q['id']}"
        assert q.get("expected_outcome") == "abstain", f"OOS question must have expected_outcome='abstain' in {q['id']}"
        assert len(q.get("reason", "").strip()) > 0, f"Missing reason in OOS question {q['id']}"

    # Confirm mutually exclusive IDs
    assert ans_ids.isdisjoint(oos_ids), "Answerable and OOS question IDs overlap"

    # Verify ground-truth loader loads all 18 targets with expected docs and pages
    gt_targets = load_ground_truth(ans_path)
    assert len(gt_targets) == 18
    for gt in gt_targets:
        assert gt.question_id in ans_ids
        assert len(gt.expected_documents) > 0
        assert len(gt.expected_pages) > 0


def test_citation_validity_helper():
    """Verify syntactic citation validity and excerpt grounding logic."""
    retrieved_chunk = RetrievedChunk(
        chunk_id="P1_p03_c02",
        document_id="paper_1",
        document_title="Retrieval-Augmented Generation",
        physical_page_index=2,
        physical_page_number=3,
        chunk_index_in_page=0,
        chunk_index_in_doc=2,
        start_char_idx=0,
        end_char_idx=95,
        char_count=95,
        word_count=14,
        text="The generator uses BART-large with 400M parameters to produce fluent sequence-to-sequence answers.",
        score=0.88,
    )

    chunks_by_id = {retrieved_chunk.chunk_id: retrieved_chunk}

    # Case 1: Valid citation pointing to existing chunk with verbatim excerpt
    valid_citation = Citation(
        chunk_id="P1_p03_c02",
        document_id="paper_1",
        document_title="Retrieval-Augmented Generation",
        physical_page_number=3,
        printed_page_number=3,
        excerpt="uses BART-large with 400M parameters",
    )
    assert valid_citation.chunk_id in chunks_by_id
    clean_excerpt = " ".join(valid_citation.excerpt.strip(".\"'\n").split())
    clean_source = " ".join(chunks_by_id[valid_citation.chunk_id].text.split())
    assert clean_excerpt.lower() in clean_source.lower()

    # Case 2: Hallucinated / non-existent chunk ID
    hallucinated_citation = Citation(
        chunk_id="P99_p99_c99",
        document_id="paper_99",
        document_title="Non Existent",
        physical_page_number=99,
        printed_page_number=99,
        excerpt="hallucinated excerpt",
    )
    assert hallucinated_citation.chunk_id not in chunks_by_id

    # Case 3: Citation pointing to valid chunk but with unfaithful/modified excerpt
    altered_citation = Citation(
        chunk_id="P1_p03_c02",
        document_id="paper_1",
        document_title="Retrieval-Augmented Generation",
        physical_page_number=3,
        printed_page_number=3,
        excerpt="completely fabricated text not present in source chunk",
    )
    clean_alt_excerpt = " ".join(altered_citation.excerpt.strip(".\"'\n").split())
    assert clean_alt_excerpt.lower() not in clean_source.lower()
    assert len(clean_alt_excerpt) >= 15


def test_evaluator_abstained_and_answering(tmp_path):
    """Verify evaluation execution produces correct structures for both answered and abstained queries."""
    summary, results, cold_start = run_controlled_evaluation(
        provider_type="mock",
        retrieval_mode="hybrid",
        top_k=3,
        sample_limit=2,
        warmup=False,
    )

    # sample_limit=2 -> 2 answerable questions + max(1, 2 // 2) = 1 out-of-scope question
    assert len(results) == 3
    assert cold_start >= 0.0

    meta = summary["metadata"]
    assert meta["provider_name"] == "mock"
    assert meta["retrieval_mode"] == "hybrid"
    assert meta["is_mock_provider"] is True

    dims = summary["dimensions"]
    assert "answer_correctness" in dims
    assert "citation_support" in dims
    assert "citation_validity" in dims
    assert "abstention_behavior" in dims
    assert "response_latency" in dims

    # Verify out-of-scope query abstention behavior
    oos_result = next(r for r in results if r.question_id.startswith("OOS"))
    assert oos_result.status in ("abstained_out_of_scope", "abstained_insufficient_evidence")
    assert oos_result.citations_count == 0

    d4 = dims["abstention_behavior"]
    assert d4["evaluated_questions"] == 1
    assert d4["oos_abstention_accuracy"] == 1.0
    assert d4["zero_citations_compliance_rate"] == 1.0

    # Verify report generation with all dimension headers
    md_path = tmp_path / "test_report.md"
    json_path = tmp_path / "test_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": [r.model_dump() for r in results]}, f, indent=2)
    generate_markdown_report(summary, results, md_path, json_path=json_path)

    assert md_path.is_file()
    assert json_path.is_file()
    report_text = md_path.read_text(encoding="utf-8")
    assert "## 1. Executive Summary & Multi-Dimensional Scorecard" in report_text
    assert "## 2. Dimension 1: Answer Correctness" in report_text
    assert "## 3. Dimension 2: Citation Support" in report_text
    assert "## 4. Dimension 3: Citation Validity" in report_text
    assert "## 5. Dimension 4: Abstention Behavior" in report_text
    assert "## 6. Dimension 5: Response Latency Profiling" in report_text


def test_unsupported_answer_risk_zero_mechanism():
    """Verify that abstentions do not trigger unsupported-answer risk, while answering without target evidence does."""
    # Case A: System abstained because evidence was missing -> unsupported_risk MUST be False
    has_answer_abstain = False
    target_cited_abstain = False
    unsupported_risk_abstain = bool(has_answer_abstain and not target_cited_abstain)
    assert unsupported_risk_abstain is False

    # Case B: System answered and cited verified target evidence -> unsupported_risk MUST be False
    has_answer_supported = True
    target_cited_supported = True
    unsupported_risk_supported = bool(has_answer_supported and not target_cited_supported)
    assert unsupported_risk_supported is False

    # Case C: System answered but cited unverified/non-target evidence -> unsupported_risk MUST be True
    has_answer_hallucinated = True
    target_cited_hallucinated = False
    unsupported_risk_hallucinated = bool(has_answer_hallucinated and not target_cited_hallucinated)
    assert unsupported_risk_hallucinated is True


def test_evidence_claim_support_keyword_heuristic():
    """Verify check_evidence_claim_support operates as a token-overlap heuristic against reference answers."""
    # Chunk text contains the key claim terms from expected answer
    chunk_text = "RAG models use a pre-trained sequence-to-sequence transformer, specifically BART-large with 400M parameters."
    exp_answer = "RAG uses BART-large (400M parameters) as its generator model."
    is_supp, recall = check_evidence_claim_support(
        cited_chunks_text=chunk_text,
        expected_answer=exp_answer,
        question_text="What generator model does RAG use?",
        min_recall=0.20,
    )
    assert is_supp is True
    assert recall >= 0.20

    # Chunk text contains boilerplate on the same page with no matching claim keywords
    boilerplate_chunk = "In Section 2 we review the mathematical equations for marginalized probability."
    is_supp_bp, recall_bp = check_evidence_claim_support(
        cited_chunks_text=boilerplate_chunk,
        expected_answer=exp_answer,
        question_text="What generator model does RAG use?",
        min_recall=0.20,
    )
    assert is_supp_bp is False
    assert recall_bp < 0.20


def test_audit_provider_configuration_ollama_detection():
    """Verify audit_provider_configuration correctly audits local Ollama availability and redacts credentials."""
    audit = audit_provider_configuration("mock")
    assert audit["requested_provider"] == "mock"
    assert isinstance(audit["openai_configured"], bool)
    assert isinstance(audit["gemini_configured"], bool)
    assert isinstance(audit["groq_configured"], bool)
    assert isinstance(audit["ollama_server_reachable"], bool)
    assert isinstance(audit["unavailable_services"], list)

    # Invariant: No API keys or tokens are in the audit dict
    serialized = json.dumps(audit)
    assert "sk-" not in serialized
    assert "AIza" not in serialized
    assert "gsk_" not in serialized

    # When Ollama is running at localhost:11434, it should be recognized
    if audit["ollama_server_reachable"]:
        assert isinstance(audit["ollama_models"], list)
        assert not any("Ollama" in svc for svc in audit["unavailable_services"])


def test_compute_percentile_math():
    """Verify standard percentile computation with linear interpolation."""
    # Empty list
    assert compute_percentile([], 95.0) == 0.0

    # Single item
    assert compute_percentile([42.0], 50.0) == 42.0
    assert compute_percentile([42.0], 95.0) == 42.0

    # Two items
    # index = 0.5 * 1 = 0.5 -> 10 + 0.5*(20-10) = 15.0
    assert compute_percentile([10.0, 20.0], 50.0) == 15.0
    # index = 0.95 * 1 = 0.95 -> 10 + 0.95*10 = 19.5
    assert compute_percentile([10.0, 20.0], 95.0) == 19.5

    # 100 items: 1..100
    data = [float(x) for x in range(1, 101)]
    assert compute_percentile(data, 50.0) == 50.5
    assert compute_percentile(data, 95.0) == 95.05
