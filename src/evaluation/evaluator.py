"""Evaluation engine computing Recall@k, MRR, and per-question retrieval diagnostics."""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from src.retrieval.retriever import DenseRetriever
from src.evaluation.models import (
    RetrievedChunkEval,
    QuestionEvaluationResult,
    EvaluationSummary,
)
from src.evaluation.ground_truth import GroundTruthTarget


class RetrievalEvaluator:
    """Evaluates dense retrieval against annotated ground truth questions."""

    def __init__(
        self,
        retriever: Optional[DenseRetriever] = None,
        default_top_k: int = 5,
    ) -> None:
        """Initialize retrieval evaluator.

        Args:
            retriever: DenseRetriever instance. If None, initialized with default config.
            default_top_k: Number of chunks to retrieve per question (default 5).
        """
        self.retriever = retriever or DenseRetriever()
        self.default_top_k = default_top_k

    def evaluate_question(
        self,
        target: GroundTruthTarget,
        top_k: Optional[int] = None,
    ) -> QuestionEvaluationResult:
        """Evaluate retrieval performance for a single ground truth question.

        Args:
            target: GroundTruthTarget with question and annotated target pages.
            top_k: Number of top chunks to retrieve.

        Returns:
            QuestionEvaluationResult with per-chunk relevance and metrics.
        """
        k = top_k or self.default_top_k
        retrieval_res = self.retriever.retrieve(query=target.question_text, top_k=k)

        eval_chunks: List[RetrievedChunkEval] = []
        first_rel_rank: Optional[int] = None
        docs_covered_at_5: List[str] = []
        unique_target_pages_retrieved: Set[Tuple[str, int]] = set()

        for rank, chunk in enumerate(retrieval_res.chunks, 1):
            is_rel = target.is_chunk_relevant(
                document_id=chunk.document_id,
                physical_page_number=chunk.physical_page_number,
            )

            # Record first relevant rank
            if is_rel and first_rel_rank is None:
                first_rel_rank = rank

            if is_rel and rank <= 5:
                if chunk.document_id not in docs_covered_at_5:
                    docs_covered_at_5.append(chunk.document_id)
                unique_target_pages_retrieved.add((chunk.document_id, chunk.physical_page_number))

            # Create clean preview snippet
            clean_snippet = " ".join(chunk.text.strip().split())
            if len(clean_snippet) > 160:
                clean_snippet = clean_snippet[:157] + "..."

            eval_chunks.append(
                RetrievedChunkEval(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    physical_page_number=chunk.physical_page_number,
                    printed_page_number=chunk.printed_page_number,
                    score=chunk.score,
                    is_relevant=is_rel,
                    snippet=clean_snippet,
                )
            )

        rr = 1.0 / first_rel_rank if first_rel_rank is not None else 0.0
        hit_1 = first_rel_rank == 1
        hit_3 = first_rel_rank is not None and first_rel_rank <= 3
        hit_5 = first_rel_rank is not None and first_rel_rank <= 5

        num_expected_docs = max(1, len(target.expected_documents))
        doc_coverage = len(docs_covered_at_5) / num_expected_docs
        all_docs_covered = len(docs_covered_at_5) == len(target.expected_documents) if target.expected_documents else False
        any_doc_covered = len(docs_covered_at_5) > 0

        # Target page recall calculation
        total_target_pages = sum(len(pages) for pages in target.expected_pages.values())
        target_pages_covered = len(unique_target_pages_retrieved)
        target_page_recall = target_pages_covered / max(1, total_target_pages)

        return QuestionEvaluationResult(
            question_id=target.question_id,
            question_text=target.question_text,
            question_type=target.question_type,
            expected_documents=target.expected_documents,
            expected_pages=target.expected_pages,
            supporting_excerpt=target.supporting_excerpt,
            retrieved_chunks=eval_chunks,
            first_relevant_rank=first_rel_rank,
            reciprocal_rank=rr,
            hit_at_1=hit_1,
            hit_at_3=hit_3,
            hit_at_5=hit_5,
            documents_covered_at_5=docs_covered_at_5,
            document_coverage_at_5=round(doc_coverage, 4),
            all_documents_covered_at_5=all_docs_covered,
            any_document_covered_at_5=any_doc_covered,
            target_pages_total=total_target_pages,
            target_pages_covered_at_5=target_pages_covered,
            target_page_recall_at_5=round(target_page_recall, 4),
        )

    def evaluate_all(
        self,
        targets: List[GroundTruthTarget],
        total_corpus_questions: int = 18,
        excluded_count: int = 0,
        top_k: Optional[int] = None,
    ) -> Tuple[EvaluationSummary, List[QuestionEvaluationResult]]:
        """Evaluate dense retrieval across all ground truth questions.

        Args:
            targets: List of GroundTruthTarget instances.
            total_corpus_questions: Total number of questions in benchmark.
            excluded_count: Count of excluded questions (e.g. out-of-scope).
            top_k: Top-k cutoff.

        Returns:
            Tuple of (EvaluationSummary, List[QuestionEvaluationResult]).
        """
        k = top_k or self.default_top_k
        results: List[QuestionEvaluationResult] = []

        for target in targets:
            q_res = self.evaluate_question(target, top_k=k)
            results.append(q_res)

        n = len(results)
        if n == 0:
            summary = EvaluationSummary(
                total_questions=total_corpus_questions,
                evaluated_questions=0,
                excluded_questions=excluded_count,
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                hit_at_1=0.0,
                hit_at_3=0.0,
                hit_at_5=0.0,
                mrr=0.0,
                mrr_at_5=0.0,
                reciprocal_rank_sum=0.0,
                mean_document_coverage_at_5=0.0,
                mean_target_page_recall_at_5=0.0,
                multi_doc_questions_count=0,
                multi_doc_any_coverage_at_5=0.0,
                multi_doc_full_coverage_at_5=0.0,
                multi_doc_mean_coverage_at_5=0.0,
            )
            return summary, results

        r1 = sum(1 for r in results if r.hit_at_1) / n
        r3 = sum(1 for r in results if r.hit_at_3) / n
        r5 = sum(1 for r in results if r.hit_at_5) / n
        raw_rrs = [(1.0 / r.first_relevant_rank) if r.first_relevant_rank is not None else 0.0 for r in results]
        rr_sum = sum(raw_rrs)
        mrr = rr_sum / n
        mean_cov = sum(r.document_coverage_at_5 for r in results) / n
        mean_page_recall = sum(r.target_page_recall_at_5 for r in results) / n

        multi_doc_results = [r for r in results if len(r.expected_documents) > 1]
        multi_doc_count = len(multi_doc_results)
        multi_any = (
            sum(1 for r in multi_doc_results if r.any_document_covered_at_5) / multi_doc_count
            if multi_doc_count > 0
            else 0.0
        )
        multi_full = (
            sum(1 for r in multi_doc_results if r.all_documents_covered_at_5) / multi_doc_count
            if multi_doc_count > 0
            else 0.0
        )
        multi_mean = (
            sum(r.document_coverage_at_5 for r in multi_doc_results) / multi_doc_count
            if multi_doc_count > 0
            else 0.0
        )

        missing_in_top_5 = [r.question_id for r in results if not r.hit_at_5]

        # Breakdown by question category
        by_type: Dict[str, List[QuestionEvaluationResult]] = defaultdict(list)
        for r in results:
            # Normalize type string (e.g. 'F (Factual)' -> 'Factual')
            t_key = r.question_type.split("(")[-1].replace(")", "").strip()
            if not t_key:
                t_key = r.question_type
            by_type[t_key].append(r)

        metrics_by_type: Dict[str, Dict[str, float]] = {}
        for t_name, t_results in by_type.items():
            m_len = len(t_results)
            t_raw_rrs = [(1.0 / r.first_relevant_rank) if r.first_relevant_rank is not None else 0.0 for r in t_results]
            metrics_by_type[t_name] = {
                "count": float(m_len),
                "recall_at_1": round(sum(1 for r in t_results if r.hit_at_1) / m_len, 4),
                "recall_at_3": round(sum(1 for r in t_results if r.hit_at_3) / m_len, 4),
                "recall_at_5": round(sum(1 for r in t_results if r.hit_at_5) / m_len, 4),
                "mrr": round(sum(t_raw_rrs) / m_len, 4),
                "mrr_at_5": round(sum(t_raw_rrs) / m_len, 4),
            }

        summary = EvaluationSummary(
            total_questions=total_corpus_questions,
            evaluated_questions=n,
            excluded_questions=excluded_count,
            recall_at_1=round(r1, 4),
            recall_at_3=round(r3, 4),
            recall_at_5=round(r5, 4),
            hit_at_1=round(r1, 4),
            hit_at_3=round(r3, 4),
            hit_at_5=round(r5, 4),
            mrr=round(mrr, 4),
            mrr_at_5=round(mrr, 4),
            reciprocal_rank_sum=round(rr_sum, 4),
            mean_document_coverage_at_5=round(mean_cov, 4),
            mean_target_page_recall_at_5=round(mean_page_recall, 4),
            multi_doc_questions_count=multi_doc_count,
            multi_doc_any_coverage_at_5=round(multi_any, 4),
            multi_doc_full_coverage_at_5=round(multi_full, 4),
            multi_doc_mean_coverage_at_5=round(multi_mean, 4),
            metrics_by_type=metrics_by_type,
            questions_missing_in_top_5=missing_in_top_5,
        )

        return summary, results

