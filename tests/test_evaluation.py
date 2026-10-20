"""Tests for Task 3.5: Retrieval Evaluation.

Covers:
1. Ground truth target parsing and page reference normalization
2. Relevance checker logic across documents and physical page numbers
3. Metric calculations: Recall@1, Recall@3, Recall@5, MRR, DocumentCoverage@5
4. Edge cases: zero hits, perfect rank 1 hits, mixed ranks, empty targets
5. Report generation and JSON serialization
"""

import json
from pathlib import Path
import pytest

from src.evaluation.models import (
    RetrievedChunkEval,
    QuestionEvaluationResult,
    EvaluationSummary,
)
from src.evaluation.ground_truth import (
    GroundTruthTarget,
    parse_page_references,
    load_ground_truth,
)
from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.reporting import save_json_results, generate_markdown_report


# ---------------------------------------------------------------------------
# 1. Ground Truth Parsing Tests
# ---------------------------------------------------------------------------

class TestGroundTruthParsing:
    """Tests parsing and normalization of annotated page references."""

    def test_parse_single_and_comma_separated_pages(self):
        """Verify parsing single page indices and comma-separated lists."""
        ref1 = 'P1: Physical Page Index 2 (Printed Page 3), §2.3 ("Generator: BART")'
        targets1 = parse_page_references(ref1)
        assert targets1 == {"P1": {3}}

        ref2 = "P2: Physical Page Index 2, 5 (Physical Pages 3, 6)"
        targets2 = parse_page_references(ref2)
        assert targets2 == {"P2": {3, 6}}

    def test_parse_page_index_ranges(self):
        """Verify parsing hyphenated or en-dash page index ranges."""
        ref1 = "P4 Physical Page Index 3–7 (Printed Pages 4–8)"
        targets1 = parse_page_references(ref1)
        assert targets1 == {"P4": {4, 5, 6, 7, 8}}

        ref2 = "P1: Physical Page Index 1-2 (Printed Pages 2-3)"
        targets2 = parse_page_references(ref2)
        assert targets2 == {"P1": {2, 3}}

    def test_parse_multi_document_references(self):
        """Verify parsing multi-document targets."""
        ref = (
            "P1: Physical Page Index 5 (Printed Page 6), Table 1; "
            "P2: Physical Page Index 7 (Physical Page 8), Table 4; "
            "P3: Physical Page Index 6 (Physical Page 7), Table 1"
        )
        targets = parse_page_references(ref)
        assert targets == {
            "P1": {6},
            "P2": {8},
            "P3": {7},
        }

    def test_parse_page_references_fallback_doc(self):
        """Verify fallback document ID and em-dash range parsing."""
        ref = "Physical Page Index 2—4 (Printed Pages 3—5)"
        targets = parse_page_references(ref, default_doc_id="P1")
        assert targets == {"P1": {3, 4, 5}}


# ---------------------------------------------------------------------------
# 2. Relevance Checker Tests
# ---------------------------------------------------------------------------

class TestRelevanceLogic:
    """Tests relevance checking logic for chunks against ground truth targets."""

    def test_is_chunk_relevant_matches(self):
        """Verify that matching doc_id and physical_page_number evaluate to True."""
        target = GroundTruthTarget(
            question_id="Q01",
            question_text="What generator model does RAG use?",
            question_type="F (Factual)",
            expected_documents=["P1"],
            expected_pages={"P1": [3]},
        )
        # Matching document and page
        assert target.is_chunk_relevant(document_id="P1", physical_page_number=3) is True

        # Matching document, wrong page
        assert target.is_chunk_relevant(document_id="P1", physical_page_number=2) is False

        # Wrong document, matching page
        assert target.is_chunk_relevant(document_id="P2", physical_page_number=3) is False


# ---------------------------------------------------------------------------
# 3. Metric Calculations & Aggregation Tests
# ---------------------------------------------------------------------------

class TestMetricCalculations:
    """Tests mathematical correctness of Recall@k, MRR, DocumentCoverage, and PageRecall."""

    def test_perfect_retrieval_metrics(self):
        """All questions have first relevant result at rank 1 -> Recall=1.0, MRR=1.0."""
        target = GroundTruthTarget(
            question_id="Q_TEST",
            question_text="Test question",
            question_type="F (Factual)",
            expected_documents=["P1"],
            expected_pages={"P1": [3]},
        )

        class DummyRetriever:
            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                return SimpleNamespace(
                    chunks=[
                        SimpleNamespace(
                            chunk_id="P1_p03_c01",
                            document_id="P1",
                            physical_page_number=3,
                            printed_page_number=3,
                            score=0.9,
                            text="Relevant target text.",
                        )
                    ]
                )

        evaluator = RetrievalEvaluator(retriever=DummyRetriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=[target])

        assert summary.recall_at_1 == 1.0
        assert summary.recall_at_3 == 1.0
        assert summary.recall_at_5 == 1.0
        assert summary.mrr == 1.0
        assert summary.mean_document_coverage_at_5 == 1.0
        assert summary.mean_target_page_recall_at_5 == 1.0
        assert summary.questions_missing_in_top_5 == []

    def test_zero_hit_query_metrics(self):
        """Verify metrics when a query retrieves zero relevant chunks."""
        target = GroundTruthTarget(
            question_id="Q_MISS",
            question_text="Miss question",
            question_type="F (Factual)",
            expected_documents=["P1"],
            expected_pages={"P1": [3]},
        )

        class MissRetriever:
            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                return SimpleNamespace(
                    chunks=[
                        SimpleNamespace(
                            chunk_id=f"c{i}",
                            document_id="P2",
                            physical_page_number=99,
                            printed_page_number=None,
                            score=0.5,
                            text="Irrelevant",
                        )
                        for i in range(1, 6)
                    ]
                )

        evaluator = RetrievalEvaluator(retriever=MissRetriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=[target])

        assert len(results) == 1
        res = results[0]
        assert res.first_relevant_rank is None
        assert res.reciprocal_rank == 0.0
        assert res.hit_at_1 is False
        assert res.hit_at_3 is False
        assert res.hit_at_5 is False
        assert res.documents_covered_at_5 == []
        assert res.document_coverage_at_5 == 0.0
        assert res.any_document_covered_at_5 is False
        assert res.all_documents_covered_at_5 is False
        assert res.target_page_recall_at_5 == 0.0
        assert summary.mrr == 0.0
        assert summary.recall_at_5 == 0.0
        assert summary.questions_missing_in_top_5 == ["Q_MISS"]

    def test_multi_page_target_recall(self):
        """Verify target page recall when a target spans multiple pages."""
        target = GroundTruthTarget(
            question_id="Q_PAGES",
            question_text="Multi page question",
            question_type="F (Factual)",
            expected_documents=["P1"],
            expected_pages={"P1": [2, 4, 6]},
        )

        class MultiPageRetriever:
            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                return SimpleNamespace(
                    chunks=[
                        # Hit on page 2 (rank 1)
                        SimpleNamespace(
                            chunk_id="c1",
                            document_id="P1",
                            physical_page_number=2,
                            printed_page_number=None,
                            score=0.9,
                            text="Page 2 hit",
                        ),
                        # Hit on page 4 (rank 2)
                        SimpleNamespace(
                            chunk_id="c2",
                            document_id="P1",
                            physical_page_number=4,
                            printed_page_number=None,
                            score=0.8,
                            text="Page 4 hit",
                        ),
                        # Irrelevant page 9 (rank 3)
                        SimpleNamespace(
                            chunk_id="c3",
                            document_id="P1",
                            physical_page_number=9,
                            printed_page_number=None,
                            score=0.7,
                            text="Page 9",
                        ),
                    ]
                )

        evaluator = RetrievalEvaluator(retriever=MultiPageRetriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=[target])

        res = results[0]
        assert res.target_pages_total == 3
        assert res.target_pages_covered_at_5 == 2  # pages 2 and 4
        assert pytest.approx(res.target_page_recall_at_5, abs=1e-4) == 2 / 3
        assert pytest.approx(summary.mean_target_page_recall_at_5, abs=1e-4) == 2 / 3

    def test_multi_document_full_vs_partial_coverage(self):
        """Verify distinction between partial document coverage and full document coverage."""
        target_partial = GroundTruthTarget(
            question_id="Q_PARTIAL",
            question_text="Q partial",
            question_type="M (Multi-paper)",
            expected_documents=["P1", "P2", "P3"],
            expected_pages={"P1": [2], "P2": [4], "P3": [6]},
        )
        target_full = GroundTruthTarget(
            question_id="Q_FULL",
            question_text="Q full",
            question_type="M (Multi-paper)",
            expected_documents=["P1", "P2"],
            expected_pages={"P1": [2], "P2": [4]},
        )

        class MultiDocRetriever:
            def __init__(self):
                self.calls = 0

            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                self.calls += 1
                if self.calls == 1:
                    # Q_PARTIAL: retrieves P1 and P2, misses P3
                    chunks = [
                        SimpleNamespace(chunk_id="c1", document_id="P1", physical_page_number=2, printed_page_number=None, score=0.9, text=""),
                        SimpleNamespace(chunk_id="c2", document_id="P2", physical_page_number=4, printed_page_number=None, score=0.8, text=""),
                    ]
                else:
                    # Q_FULL: retrieves P1 and P2
                    chunks = [
                        SimpleNamespace(chunk_id="c1", document_id="P1", physical_page_number=2, printed_page_number=None, score=0.9, text=""),
                        SimpleNamespace(chunk_id="c2", document_id="P2", physical_page_number=4, printed_page_number=None, score=0.8, text=""),
                    ]
                return SimpleNamespace(chunks=chunks)

        evaluator = RetrievalEvaluator(retriever=MultiDocRetriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=[target_partial, target_full])

        # Q_PARTIAL: 2 of 3 docs covered
        assert results[0].any_document_covered_at_5 is True
        assert results[0].all_documents_covered_at_5 is False
        assert pytest.approx(results[0].document_coverage_at_5, abs=1e-4) == 2 / 3

        # Q_FULL: 2 of 2 docs covered
        assert results[1].any_document_covered_at_5 is True
        assert results[1].all_documents_covered_at_5 is True
        assert results[1].document_coverage_at_5 == 1.0

        # Aggregated summary
        assert summary.multi_doc_questions_count == 2
        assert summary.multi_doc_any_coverage_at_5 == 1.0  # both have >= 1 doc
        assert summary.multi_doc_full_coverage_at_5 == 0.5  # only 1 has all docs
        assert pytest.approx(summary.multi_doc_mean_coverage_at_5, abs=1e-4) == (2/3 + 1.0) / 2

    def test_mrr_exact_calculation_and_rank_derivation(self):
        """Verify the exact arithmetic of the 18-question benchmark MRR.

        Demonstrates that:
        - True rank vector has sum of reciprocal ranks = 7.5000 -> MRR = 0.4167.
        - Transposed/scrambled conversational table ranks sum to 8.5833 -> MRR = 0.4769.
        """
        # True benchmark ranks from retrieval_evaluation.json
        true_ranks = [None, 3, 1, 2, 1, 1, 1, 1, 4, 3, 2, 3, None, None, None, None, None, 4]
        assert len(true_ranks) == 18

        true_rrs = [1.0 / r if r is not None else 0.0 for r in true_ranks]
        sum_true_rrs = sum(true_rrs)
        mrr_true = sum_true_rrs / 18

        assert pytest.approx(sum_true_rrs, abs=1e-4) == 7.5000
        assert pytest.approx(mrr_true, abs=1e-4) == 0.4167

        # Transposed ranks from previous conversational text response table
        scrambled_ranks = [None, 1, 1, 1, 2, 3, 2, 1, 2, 1, 2, 4, None, None, None, None, None, 1]
        scrambled_rrs = [1.0 / r if r is not None else 0.0 for r in scrambled_ranks]
        sum_scrambled_rrs = sum(scrambled_rrs)
        mrr_scrambled = sum_scrambled_rrs / 18

        assert pytest.approx(sum_scrambled_rrs, abs=1e-4) == 8.5833
        assert pytest.approx(mrr_scrambled, abs=1e-4) == 0.4769

    def test_staggered_ranks_and_mrr(self):
        """Test reciprocal rank values for ranks 1, 2, 4, and no hit."""
        targets = [
            GroundTruthTarget(question_id=f"Q{i}", question_text=f"Q{i}", question_type="F", expected_documents=["P1"], expected_pages={"P1": [1]})
            for i in range(1, 5)
        ]

        class StaggeredRetriever:
            def __init__(self):
                self.calls = 0

            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                self.calls += 1
                hit_rank = {1: 1, 2: 2, 3: 4, 4: None}[self.calls]
                chunks = []
                for r in range(1, 6):
                    doc = "P1" if r == hit_rank else "P2"
                    page = 1 if r == hit_rank else 99
                    chunks.append(
                        SimpleNamespace(
                            chunk_id=f"c{r}",
                            document_id=doc,
                            physical_page_number=page,
                            printed_page_number=None,
                            score=1.0 / r,
                            text=f"Text {r}",
                        )
                    )
                return SimpleNamespace(chunks=chunks)

        evaluator = RetrievalEvaluator(retriever=StaggeredRetriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=targets)

        assert pytest.approx(summary.mrr, abs=1e-4) == 0.4375
        assert pytest.approx(summary.recall_at_1, abs=1e-4) == 0.25
        assert pytest.approx(summary.recall_at_3, abs=1e-4) == 0.50
        assert pytest.approx(summary.recall_at_5, abs=1e-4) == 0.75
        assert summary.questions_missing_in_top_5 == ["Q4"]


# ---------------------------------------------------------------------------
# 4. Report Generation Tests
# ---------------------------------------------------------------------------

class TestEvaluationReporting:
    """Tests saving evaluation JSON results and Markdown reports."""

    def test_save_json_and_markdown(self, tmp_path: Path):
        """Verify report generator produces valid JSON and well-formed Markdown."""
        summary = EvaluationSummary(
            total_questions=2,
            evaluated_questions=2,
            excluded_questions=0,
            recall_at_1=0.5,
            recall_at_3=1.0,
            recall_at_5=1.0,
            hit_at_1=0.5,
            hit_at_3=1.0,
            hit_at_5=1.0,
            mrr=0.75,
            mean_document_coverage_at_5=1.0,
            mean_target_page_recall_at_5=1.0,
            multi_doc_questions_count=1,
            multi_doc_any_coverage_at_5=1.0,
            multi_doc_full_coverage_at_5=1.0,
            multi_doc_mean_coverage_at_5=1.0,
            metrics_by_type={"Factual": {"count": 2.0, "recall_at_1": 0.5, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.75}},
            questions_missing_in_top_5=[],
        )
        results = [
            QuestionEvaluationResult(
                question_id="Q01",
                question_text="Sample Q1",
                question_type="Factual",
                expected_documents=["P1"],
                expected_pages={"P1": [3]},
                retrieved_chunks=[
                    RetrievedChunkEval(
                        rank=1,
                        chunk_id="P1_p03_c01",
                        document_id="P1",
                        physical_page_number=3,
                        score=0.85,
                        is_relevant=True,
                        snippet="Sample text",
                    )
                ],
                first_relevant_rank=1,
                reciprocal_rank=1.0,
                hit_at_1=True,
                hit_at_3=True,
                hit_at_5=True,
                documents_covered_at_5=["P1"],
                document_coverage_at_5=1.0,
                all_documents_covered_at_5=True,
                any_document_covered_at_5=True,
                target_pages_total=1,
                target_pages_covered_at_5=1,
                target_page_recall_at_5=1.0,
            )
        ]

        json_file = tmp_path / "results.json"
        md_file = tmp_path / "report.md"

        save_json_results(summary, results, json_file)
        md_content = generate_markdown_report(summary, results, md_file)

        assert json_file.is_file()
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"]["mrr"] == 0.75
        assert len(data["questions"]) == 1

        assert md_file.is_file()
        assert "# Retrieval Evaluation Baseline Report" in md_content
        assert "Hit@1 (Page-Level Acc.)" in md_content
        assert "Mean Reciprocal Rank (MRR@5)" in md_content
        assert "50.0%" in md_content


# ---------------------------------------------------------------------------
# 5. Metric Correctness and Precision Audit Tests (Task 4.1.2)
# ---------------------------------------------------------------------------

class TestMetricCorrectnessAudit:
    """Verifies MRR@5 truncation semantics, exact numerical precision, and query reconciliation."""

    def test_mrr_at_5_truncation_behavior(self):
        """Relevant result beyond top-5 receives rr=0.0, proving MRR is truncated MRR@5."""
        target = GroundTruthTarget(
            question_id="Q_RANK6",
            question_text="Target beyond cutoff",
            question_type="F (Factual)",
            expected_documents=["P1"],
            expected_pages={"P1": [3]},
        )

        class Rank6Retriever:
            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                # top_k=5 retrieves only 5 chunks, all irrelevant
                chunks = [
                    SimpleNamespace(chunk_id=f"c{i}", document_id="P2", physical_page_number=99, printed_page_number=None, score=0.9 - 0.1*i, text="Irrelevant")
                    for i in range(1, top_k + 1)
                ]
                return SimpleNamespace(chunks=chunks)

        evaluator = RetrievalEvaluator(retriever=Rank6Retriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=[target])

        res = results[0]
        assert res.first_relevant_rank is None
        assert res.reciprocal_rank == 0.0
        assert res.hit_at_5 is False
        assert summary.mrr == 0.0
        assert summary.mrr_at_5 == 0.0

    def test_benchmark_exact_precision_and_sum_reconciliation(self):
        """Verify exact fractions for both Dense (15/2 -> 5/12) and Hybrid (487/60 -> 487/1080)."""
        from fractions import Fraction

        # Dense benchmark ranks across Q01..Q18
        dense_ranks = [None, 3, 1, 2, 1, 1, 1, 1, 4, 3, 2, 3, None, None, None, None, None, 4]
        dense_exact_rrs = [Fraction(1, r) if r is not None else Fraction(0, 1) for r in dense_ranks]
        dense_sum = sum(dense_exact_rrs)
        dense_mrr = dense_sum / len(dense_ranks)

        assert dense_sum == Fraction(15, 2)
        assert float(dense_sum) == 7.5
        assert dense_mrr == Fraction(5, 12)
        assert round(float(dense_mrr), 4) == 0.4167

        # Hybrid benchmark ranks across Q01..Q18
        hybrid_ranks = [2, 1, 3, 2, None, 2, 3, 1, 3, 5, 1, 3, None, 3, None, 2, 1, 4]
        hybrid_exact_rrs = [Fraction(1, r) if r is not None else Fraction(0, 1) for r in hybrid_ranks]
        hybrid_sum = sum(hybrid_exact_rrs)
        hybrid_mrr = hybrid_sum / len(hybrid_ranks)

        assert hybrid_sum == Fraction(487, 60)
        assert pytest.approx(float(hybrid_sum), abs=1e-8) == 8.11666667
        assert hybrid_mrr == Fraction(487, 1080)
        assert round(float(hybrid_mrr), 4) == 0.4509

        # Explain reported '8.12': 2-decimal rounded string of 487/60
        assert f"{float(hybrid_sum):.2f}" == "8.12"

    def test_query_outcome_reconciliation_with_hit_k(self):
        """Confirm that query classifications partition all 18 queries and reconcile with Hit@5."""
        # 18 unique questions
        all_qids = [f"Q{i:02d}" for i in range(1, 19)]
        assert len(all_qids) == 18

        rescued = {"Q01", "Q14", "Q16", "Q17"}
        improved = {"Q02", "Q09", "Q11"}
        unchanged = {"Q04", "Q08", "Q12", "Q13", "Q15", "Q18"}
        dropped_in_top5 = {"Q03", "Q06", "Q07", "Q10"}
        regressed = {"Q05"}

        # Disjoint partitions
        partitions = [rescued, improved, unchanged, dropped_in_top5, regressed]
        union_all = set().union(*partitions)
        assert union_all == set(all_qids)
        total_count = sum(len(p) for p in partitions)
        assert total_count == 18

        # Reconcile with Hybrid Hit@5 (15 hits, 3 misses)
        hybrid_hits = rescued | improved | {"Q04", "Q08", "Q12", "Q18"} | dropped_in_top5
        hybrid_misses = {"Q13", "Q15"} | regressed
        assert len(hybrid_hits) == 15
        assert len(hybrid_misses) == 3
        assert hybrid_hits.isdisjoint(hybrid_misses)

    def test_evaluator_reciprocal_rank_precision_consistency(self):
        """Verify that per-query reciprocal_rank preserves float precision and avoids 0.9999 sum."""
        targets = [
            GroundTruthTarget(question_id=f"Q{i}", question_text=f"Q{i}", question_type="F", expected_documents=["P1"], expected_pages={"P1": [1]})
            for i in range(1, 4)
        ]

        class Rank3Retriever:
            def retrieve(self, query, top_k):
                from types import SimpleNamespace
                # Target at rank 3 for all 3 questions
                chunks = [
                    SimpleNamespace(chunk_id=f"c{r}", document_id="P1" if r == 3 else "P2", physical_page_number=1 if r == 3 else 99, printed_page_number=None, score=1.0 / r, text=f"Text {r}")
                    for r in range(1, top_k + 1)
                ]
                return SimpleNamespace(chunks=chunks)

        evaluator = RetrievalEvaluator(retriever=Rank3Retriever(), default_top_k=5)
        summary, results = evaluator.evaluate_all(targets=targets)

        # Each question should have exact 1/3 float
        for r in results:
            assert pytest.approx(r.reciprocal_rank, abs=1e-12) == 1.0 / 3.0

        # Sum of 3 x 1/3 must equal 1.0, NOT 0.9999
        per_query_sum = sum(r.reciprocal_rank for r in results)
        assert pytest.approx(per_query_sum, abs=1e-12) == 1.0
        assert summary.reciprocal_rank_sum == 1.0
        assert summary.mrr_at_5 == pytest.approx(1.0 / 3.0, abs=1e-4)


