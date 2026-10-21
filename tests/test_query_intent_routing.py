"""Focused tests for Task 4.4.19: Lightweight Query-Intent Routing.

Tests cover:
1. Clear comparative queries in multiple phrasings ('compare', 'versus', 'differ from', 'relate to').
2. Ordinary factual questions with multi-part concepts or 'and' routing to STANDARD.
3. Intra-paper comparisons (e.g. RAG-Sequence vs RAG-Token) conservatively routing to STANDARD.
4. Ambiguous and empty queries safely defaulting to STANDARD.
5. HybridRetriever dynamic configuration selection based on detected intent.
6. Baseline behavior strictly preserved when intent routing is disabled (default).
7. Zero runtime ground-truth leakage or dependency on target labels.
8. End-to-end integration via ResearchEngineService.
"""

import pytest
from typing import List

from src.retrieval.router import (
    QueryIntent,
    QueryIntentResult,
    classify_query_intent,
)
from src.retrieval.models import (
    RetrievedChunk,
    HybridRetrievalConfig,
)
from src.retrieval.hybrid import HybridRetriever
from src.ui.service import ResearchEngineService
from src.generation.models import AnswerStatus


def make_dummy_chunk(chunk_id: str, doc_id: str, page: int = 1, score: float = 0.5) -> RetrievedChunk:
    """Create dummy chunk for lightweight unit testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_title=f"Paper {doc_id}",
        physical_page_index=page - 1,
        physical_page_number=page,
        printed_page_number=page,
        chunk_index_in_page=0,
        chunk_index_in_doc=0,
        start_char_idx=0,
        end_char_idx=50,
        char_count=50,
        word_count=10,
        text=f"Sample text for {chunk_id}",
        score=score,
    )


class TestQueryIntentClassifier:
    """Unit tests for classify_query_intent deterministic rule classifier."""

    @pytest.mark.parametrize(
        "query, expected_entities",
        [
            (
                "How do the Extractive Reader in DPR and Generative RAG compare on TriviaQA exact match performance?",
                {"DPR", "RAG"},
            ),
            (
                "How do REALM and T5-11B+SSM compare on WebQuestions exact match?",
                {"REALM", "T5"},
            ),
            (
                "How does Self-RAG's adaptive retrieval compare to standard RAG's unconditional always-retrieve approach?",
                {"Self-RAG", "RAG"},
            ),
            (
                "What exact match scores does RAG-Token achieve on Natural Questions, and how does this compare to DPR's extractive approach?",
                {"RAG", "DPR"},
            ),
            (
                "How do RAG, DPR, and REALM compare on Natural Questions exact match?",
                {"RAG", "DPR", "REALM"},
            ),
            (
                "How does REALM's index refresh strategy differ from DPR's static index?",
                {"REALM", "DPR"},
            ),
            (
                "How does Self-RAG's approach to deciding when to retrieve differ from standard RAG?",
                {"Self-RAG", "RAG"},
            ),
            (
                "How does Self-RAG's ISSUP token relate to RAGAS's faithfulness metric?",
                {"Self-RAG", "RAGAS"},
            ),
            (
                "Do RAG and REALM use the same knowledge corpus, and how do their passage sizes compare?",
                {"RAG", "REALM"},
            ),
            (
                "Dense Passage Retrieval versus BM25 on open-domain question answering",
                {"DPR", "BM25"},
            ),
        ],
    )
    def test_clear_comparative_queries_route_to_comparative(self, query: str, expected_entities: set):
        """Cross-paper comparative queries with explicit cues and distinct entities route to COMPARATIVE."""
        res = classify_query_intent(query)
        assert res.intent == QueryIntent.COMPARATIVE
        assert len(res.cues_matched) > 0
        assert expected_entities.issubset(set(res.entities_matched))
        assert res.is_multi_entity is True

    @pytest.mark.parametrize(
        "query",
        [
            "What generator model does RAG use?",
            "How does DPR train its passage encoder, and what types of negatives are used?",
            "What are the four reflection tokens in Self-RAG and what does each one assess?",
            "How does RAGAS compute its faithfulness metric?",
            "What role does salient span masking play in REALM, and what happens without it?",
            "What are the known limitations of using LLMs as evaluators in RAGAS?",
            "What base language models are used across the five papers?",
            "How does Self-RAG's critic model get its training data?",
            "What limitations does the RAG paper acknowledge about its static document index?",
            "What evaluation benchmarks are shared across the 2020 papers, and which are unique to each?",
        ],
    )
    def test_factual_and_multipart_questions_route_to_standard(self, query: str):
        """Factual single-paper queries and multi-part queries with 'and' route to STANDARD."""
        res = classify_query_intent(query)
        assert res.intent == QueryIntent.STANDARD

    def test_intra_paper_comparison_conservatively_routes_to_standard(self):
        """Comparisons within a single paper family (e.g. RAG-Sequence vs RAG-Token) route to STANDARD."""
        q = "What is the difference between RAG-Sequence and RAG-Token?"
        res = classify_query_intent(q)
        assert res.intent == QueryIntent.STANDARD
        assert "difference between" in res.cues_matched
        assert res.entities_matched == ["RAG"]
        assert "intra-paper" in res.reason.lower() or "single system" in res.reason.lower()

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   \t\n  ",
            "What is the difference?",
            "How do these work?",
            "Tell me about transformer attention mechanisms",
        ],
    )
    def test_ambiguous_and_empty_queries_default_to_standard(self, query: str):
        """Ambiguous comparison without named systems or empty queries conservatively default to STANDARD."""
        res = classify_query_intent(query)
        assert res.intent == QueryIntent.STANDARD

    def test_no_ground_truth_cheating(self):
        """Classifier depends only on query string, without taking target paper IDs or datasets."""
        import inspect
        sig = inspect.signature(classify_query_intent)
        params = list(sig.parameters.keys())
        assert params == ["query"]


class TestHybridRetrieverIntentRouting:
    """Tests HybridRetriever's dynamic routing integration."""

    def test_config_defaults(self):
        """Routing is opt-in (disabled by default) in HybridRetrievalConfig."""
        config = HybridRetrievalConfig()
        assert config.enable_intent_routing is False
        assert config.comparative_max_chunks_per_doc == 2

    def test_routing_disabled_by_default_preserves_baseline(self):
        """When intent routing is not enabled, query_intent is None and baseline RRF applies."""
        config = HybridRetrievalConfig(enable_intent_routing=False)
        retriever = HybridRetriever(config=config)
        try:
            # Query is comparative, but routing is disabled
            res = retriever.retrieve(
                query="How do RAG and DPR compare on TriviaQA?",
                top_k=5,
            )
            assert res.query_intent is None
        finally:
            retriever.close()

    def test_routing_active_sets_comparative_cap(self):
        """When intent routing is enabled, comparative query gets cap=2 and records intent."""
        config = HybridRetrievalConfig(enable_intent_routing=True, comparative_max_chunks_per_doc=2)
        retriever = HybridRetriever(config=config)
        try:
            res = retriever.retrieve(
                query="How do RAG and DPR compare on TriviaQA exact match?",
                top_k=5,
            )
            assert res.query_intent == "COMPARATIVE"
            # Verify document diversification took effect (no single doc monopolizes all 5 slots)
            doc_counts = {}
            for c in res.chunks:
                doc_counts[c.document_id] = doc_counts.get(c.document_id, 0) + 1
            for doc, count in doc_counts.items():
                assert count <= 2 or res.total_retrieved < 5
        finally:
            retriever.close()

    def test_routing_active_standard_query_unconstrained(self):
        """When intent routing is enabled, standard query gets unconstrained baseline ranking."""
        config = HybridRetrievalConfig(enable_intent_routing=True)
        retriever = HybridRetriever(config=config)
        try:
            res = retriever.retrieve(
                query="What generator model does RAG use?",
                top_k=5,
            )
            assert res.query_intent == "STANDARD"
        finally:
            retriever.close()


class TestServiceIntentRoutingIntegration:
    """Tests ResearchEngineService with opt-in intent routing."""

    def test_service_ask_question_with_routing_mock(self):
        """Service executes end-to-end grounded Q&A with intent routing enabled."""
        service = ResearchEngineService()
        try:
            ans = service.ask_question(
                question="How do RAG and DPR compare on Natural Questions exact match?",
                provider_name="mock",
                enable_intent_routing=True,
                top_k=5,
            )
            assert ans.status in (
                AnswerStatus.ANSWERED,
                AnswerStatus.PARTIALLY_ANSWERED,
                AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE,
            )
            assert isinstance(ans.retrieved_chunks, list)
        finally:
            service.close()

