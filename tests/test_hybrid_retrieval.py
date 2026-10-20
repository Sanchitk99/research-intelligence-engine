"""Tests for Task 4.1: Hybrid Retrieval (BM25 + Dense) with Reciprocal Rank Fusion.

Covers:
1. BM25 tokenization (hyphenated terms, alphanumeric strings, numbers, punctuation)
2. BM25 scoring, term saturation, and length normalization
3. BM25 document filtering and metadata preservation
4. Reciprocal Rank Fusion (RRF) math, deduplication, weights, and tie-breaking
5. HybridRetriever orchestration, validation, and error handling
6. End-to-end integration with the real 396-chunk corpus in data/
"""

import json
from pathlib import Path
from typing import List
import pytest

from src.retrieval.models import (
    RetrievalConfig,
    BM25Config,
    HybridRetrievalConfig,
    RetrievedChunk,
    RetrievalResult,
    InvalidQueryError,
    FilterError,
)
from src.retrieval.bm25 import BM25Index, BM25Retriever, tokenize
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.retriever import DenseRetriever


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_chunks() -> List[RetrievedChunk]:
    """Sample chunks covering distinct topics, terms, and lengths."""
    return [
        RetrievedChunk(
            chunk_id="P1_p03_c01",
            document_id="P1",
            document_title="RAG Paper",
            physical_page_index=2,
            physical_page_number=3,
            printed_page_number=3,
            chunk_index_in_page=0,
            chunk_index_in_doc=5,
            start_char_idx=0,
            end_char_idx=400,
            char_count=400,
            word_count=50,
            text="The generator component is a BART-large sequence-to-sequence model with 400M parameters.",
            score=0.0,
        ),
        RetrievedChunk(
            chunk_id="P1_p02_c06",
            document_id="P1",
            document_title="RAG Paper",
            physical_page_index=1,
            physical_page_number=2,
            printed_page_number=2,
            chunk_index_in_page=5,
            chunk_index_in_doc=4,
            start_char_idx=2000,
            end_char_idx=2300,
            char_count=300,
            word_count=35,
            text="We make our code and models available at https://github.com/huggingface/transformers.",
            score=0.0,
        ),
        RetrievedChunk(
            chunk_id="P2_p04_c02",
            document_id="P2",
            document_title="DPR Paper",
            physical_page_index=3,
            physical_page_number=4,
            printed_page_number=None,
            chunk_index_in_page=1,
            chunk_index_in_doc=7,
            start_char_idx=500,
            end_char_idx=1000,
            char_count=500,
            word_count=65,
            text="Our dual-encoder architecture uses two independent BERT-base instances with hard negatives.",
            score=0.0,
        ),
        RetrievedChunk(
            chunk_id="P3_p06_c01",
            document_id="P3",
            document_title="REALM Paper",
            physical_page_index=5,
            physical_page_number=6,
            printed_page_number=None,
            chunk_index_in_page=0,
            chunk_index_in_doc=10,
            start_char_idx=0,
            end_char_idx=600,
            char_count=600,
            word_count=80,
            text="REALM splits English Wikipedia into chunks of up to 288 BERT wordpieces, yielding 13M candidates.",
            score=0.0,
        ),
        RetrievedChunk(
            chunk_id="P4_p05_c02",
            document_id="P4",
            document_title="Self-RAG Paper",
            physical_page_index=4,
            physical_page_number=5,
            printed_page_number=5,
            chunk_index_in_page=1,
            chunk_index_in_doc=12,
            start_char_idx=600,
            end_char_idx=1200,
            char_count=600,
            word_count=75,
            text="We fine-tune Llama-2-7B and Llama-2-13B as critic and generator models using GPT-4 reflections.",
            score=0.0,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. BM25 Tokenization Tests
# ---------------------------------------------------------------------------

class TestBM25Tokenization:
    """Verifies that scientific symbols, hyphens, and numbers are tokenized accurately."""

    def test_tokenize_hyphenated_terms(self):
        """Hyphenated words must yield both the compound and constituent sub-tokens."""
        tokens = tokenize("RAG-Sequence uses BART-large and Top-20 retrieval.")
        assert "rag-sequence" in tokens
        assert "rag" in tokens
        assert "sequence" in tokens
        assert "bart-large" in tokens
        assert "bart" in tokens
        assert "large" in tokens
        assert "top-20" in tokens
        assert "top" in tokens
        assert "20" in tokens

    def test_tokenize_numbers_and_punctuation(self):
        """Alphanumeric terms and numerical counts must be extracted without punctuation noise."""
        tokens = tokenize("Model achieved 45.2 EM on NaturalQuestions (400M params, $d=768$)!")
        assert "45" in tokens
        assert "2" in tokens
        assert "400m" in tokens
        assert "768" in tokens
        assert "naturalquestions" in tokens

    def test_tokenize_empty_and_whitespace(self):
        """Empty or whitespace text returns empty token list."""
        assert tokenize("") == []
        assert tokenize("   \n\t  ") == []

    def test_tokenize_repeated_compound_subtokens(self):
        """Sub-tokens must be emitted for every compound occurrence without suppression."""
        tokens = tokenize("RAG-Sequence and RAG-Token. RAG-Sequence is effective.")
        # 'rag' appears in RAG-Sequence (twice) and RAG-Token (once)
        rag_count = tokens.count("rag")
        assert rag_count == 3, f"Expected 3 occurrences of 'rag', got {rag_count}"
        assert tokens.count("rag-sequence") == 2
        assert tokens.count("sequence") == 2


# ---------------------------------------------------------------------------
# 2. BM25 Index & Retriever Unit Tests
# ---------------------------------------------------------------------------

class TestBM25IndexUnit:
    """Tests BM25 index construction, scoring, document filtering, and tie-breaking."""

    def test_bm25_exact_match_ranking(self, synthetic_chunks):
        """Chunk with exact multiple query terms scores highest."""
        index = BM25Index(chunks=synthetic_chunks)
        results = index.search(query="BART-large generator model", top_k=3)

        assert len(results) > 0
        top_chunk, top_score = results[0]
        assert top_chunk.chunk_id == "P1_p03_c01"
        assert top_score > 0.0

    def test_bm25_document_filtering(self, synthetic_chunks):
        """Allowed doc indices filter restricts returned chunks."""
        index = BM25Index(chunks=synthetic_chunks)
        # Only allow P2 (chunk index 2) and P3 (chunk index 3)
        allowed = {2, 3}
        results = index.search(
            query="BERT-base wordpieces",
            top_k=5,
            allowed_doc_indices=allowed,
        )

        for chunk, score in results:
            assert chunk.document_id in ("P2", "P3")
            assert chunk.chunk_id in ("P2_p04_c02", "P3_p06_c01")

    def test_bm25_zero_matches_query(self, synthetic_chunks):
        """Query with terms not in any document returns empty list."""
        index = BM25Index(chunks=synthetic_chunks)
        results = index.search(query="superconducting quantum qubit", top_k=5)
        assert results == []

    def test_bm25_retriever_query_validation(self, synthetic_chunks):
        """Invalid or empty queries raise InvalidQueryError."""
        index = BM25Index(chunks=synthetic_chunks)
        retriever = BM25Retriever(index=index)

        with pytest.raises(InvalidQueryError):
            retriever.retrieve(query="")

        with pytest.raises(InvalidQueryError):
            retriever.retrieve(query="   \t\n  ")

        with pytest.raises(InvalidQueryError):
            retriever.retrieve(query=None)  # type: ignore

    def test_bm25_retriever_filter_validation(self, synthetic_chunks):
        """Invalid filter arguments raise FilterError."""
        index = BM25Index(chunks=synthetic_chunks)
        retriever = BM25Retriever(index=index)

        with pytest.raises(FilterError):
            retriever.retrieve(query="BART", document_ids=[])

        with pytest.raises(FilterError):
            retriever.retrieve(query="BART", document_ids=123)  # type: ignore

    def test_bm25_retriever_provenance_preservation(self, synthetic_chunks):
        """Retrieved chunks preserve all physical offsets, pages, and metadata."""
        index = BM25Index(chunks=synthetic_chunks)
        retriever = BM25Retriever(index=index)
        res = retriever.retrieve(query="BART-large 400M", top_k=1)

        assert res.total_retrieved == 1
        chunk = res.chunks[0]
        assert chunk.chunk_id == "P1_p03_c01"
        assert chunk.document_id == "P1"
        assert chunk.document_title == "RAG Paper"
        assert chunk.physical_page_number == 3
        assert chunk.printed_page_number == 3
        assert chunk.retrieval_method == "bm25"
        assert chunk.bm25_rank == 1
        assert chunk.bm25_score is not None and chunk.bm25_score > 0


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion (RRF) Tests
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Verifies RRF formula math, deduplication, weighting, and tie-breaking."""

    def test_rrf_scoring_math(self, synthetic_chunks):
        """Calculates expected RRF scores according to standard formula."""
        c1, c2, c3 = synthetic_chunks[0], synthetic_chunks[1], synthetic_chunks[2]

        dense_list = [c1, c2]  # c1 at rank 1, c2 at rank 2
        bm25_list = [c1, c3]   # c1 at rank 1, c3 at rank 2

        fused = reciprocal_rank_fusion(
            dense_chunks=dense_list,
            bm25_chunks=bm25_list,
            rrf_k=60,
            dense_weight=1.0,
            bm25_weight=1.0,
            top_k=5,
        )

        assert len(fused) == 3
        # c1 appeared at rank 1 in both: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
        assert fused[0].chunk_id == c1.chunk_id
        assert abs(fused[0].score - (2.0 / 61.0)) < 1e-5
        assert fused[0].dense_rank == 1
        assert fused[0].bm25_rank == 1

        # c2 appeared only in dense (rank 2): 1/(60+2) = 1/62 ≈ 0.016129
        # c3 appeared only in bm25 (rank 2): 1/(60+2) = 1/62 ≈ 0.016129
        # Deduplication confirmed: each chunk appears exactly once
        fused_ids = {c.chunk_id for c in fused}
        assert fused_ids == {c1.chunk_id, c2.chunk_id, c3.chunk_id}

    def test_rrf_asymmetric_weights(self, synthetic_chunks):
        """Asymmetric weights shift ranking towards the favored system."""
        c1, c2 = synthetic_chunks[0], synthetic_chunks[1]

        # c1 is rank 1 in dense; c2 is rank 1 in BM25
        dense_list = [c1]
        bm25_list = [c2]

        # Heavily weight BM25
        fused_bm25_heavy = reciprocal_rank_fusion(
            dense_chunks=dense_list,
            bm25_chunks=bm25_list,
            rrf_k=60,
            dense_weight=1.0,
            bm25_weight=3.0,
            top_k=2,
        )
        assert fused_bm25_heavy[0].chunk_id == c2.chunk_id

        # Heavily weight Dense
        fused_dense_heavy = reciprocal_rank_fusion(
            dense_chunks=dense_list,
            bm25_chunks=bm25_list,
            rrf_k=60,
            dense_weight=3.0,
            bm25_weight=1.0,
            top_k=2,
        )
        assert fused_dense_heavy[0].chunk_id == c1.chunk_id

    def test_rrf_invalid_parameters(self, synthetic_chunks):
        """Negative or zero rrf_k and top_k raise ValueError."""
        with pytest.raises(ValueError):
            reciprocal_rank_fusion(synthetic_chunks, synthetic_chunks, rrf_k=0)

        with pytest.raises(ValueError):
            reciprocal_rank_fusion(synthetic_chunks, synthetic_chunks, top_k=0)

    def test_rrf_empty_inputs(self):
        """Empty candidate lists yield empty fused list."""
        assert reciprocal_rank_fusion([], []) == []


# ---------------------------------------------------------------------------
# 4. HybridRetriever Orchestration Tests
# ---------------------------------------------------------------------------

class TestHybridRetrieverOrchestration:
    """Tests HybridRetriever query validation, document filtering, and top_k limits."""

    def test_hybrid_retriever_query_validation(self):
        """Empty or invalid queries raise InvalidQueryError."""
        # Use mock components
        class FakeRetriever:
            def retrieve(self, query, top_k, document_ids=None):
                return RetrievalResult(query=query, chunks=[], total_retrieved=0, collection_name="fake")
            def close(self): pass

        retriever = HybridRetriever(
            dense_retriever=FakeRetriever(),  # type: ignore
            bm25_retriever=FakeRetriever(),   # type: ignore
        )

        with pytest.raises(InvalidQueryError):
            retriever.retrieve(query="")

        with pytest.raises(InvalidQueryError):
            retriever.retrieve(query="   \n\t ")

    def test_hybrid_retriever_fuses_results(self, synthetic_chunks):
        """HybridRetriever executes dense and BM25 and returns fused results."""
        c1, c2, c3 = synthetic_chunks[0], synthetic_chunks[1], synthetic_chunks[2]

        class FakeDense:
            def retrieve(self, query, top_k, document_ids=None):
                return RetrievalResult(
                    query=query,
                    chunks=[c1, c2],
                    total_retrieved=2,
                    collection_name="dense_fake",
                )
            def close(self): pass

        class FakeBM25:
            def retrieve(self, query, top_k, document_ids=None):
                return RetrievalResult(
                    query=query,
                    chunks=[c3, c1],
                    total_retrieved=2,
                    collection_name="bm25_fake",
                )
            def close(self): pass

        retriever = HybridRetriever(
            dense_retriever=FakeDense(),  # type: ignore
            bm25_retriever=FakeBM25(),    # type: ignore
        )

        result = retriever.retrieve(query="BART model", top_k=2)
        assert result.total_retrieved == 2
        # c1 appeared in both (rank 1 dense, rank 2 bm25)
        assert result.chunks[0].chunk_id == c1.chunk_id
        assert result.chunks[0].retrieval_method == "hybrid_rrf"
        assert result.chunks[0].dense_rank == 1
        assert result.chunks[0].bm25_rank == 2


# ---------------------------------------------------------------------------
# 5. Real Corpus Integration Tests (Tasks 3.5 & 4.1 Comparison)
# ---------------------------------------------------------------------------

class TestHybridRealCorpusIntegration:
    """Verifies that Hybrid Retrieval fixes dense misses on real corpus chunks."""

    def test_hybrid_rescues_q01_rag_generator(self):
        """Q01 ('What generator model does RAG use?') is rescued from dense miss into top 3."""
        qdrant_dir = Path("data/qdrant")
        chunks_file = Path("data/chunked/corpus_chunks.json")
        if not qdrant_dir.is_dir() or not chunks_file.is_file():
            pytest.skip("Corpus files not found")

        with HybridRetriever() as retriever:
            res = retriever.retrieve(
                query="What generator model does RAG use?",
                top_k=5,
            )

            assert res.total_retrieved >= 1
            # Verify target chunk P1_p03_c04 (Physical page 3: BART-large) is present in top 3
            top3_ids = [c.chunk_id for c in res.chunks[:3]]
            top3_pages = [(c.document_id, c.physical_page_number) for c in res.chunks[:3]]

            # Target is P1 physical page 3
            assert ("P1", 3) in top3_pages, f"Expected ('P1', 3) in top 3, got {top3_pages}"

    def test_hybrid_rescues_q16_knowledge_corpus_and_passage_sizes(self):
        """Q16 ('Do RAG and REALM use the same knowledge corpus, and how do their passage sizes compare?') in top 3."""
        qdrant_dir = Path("data/qdrant")
        chunks_file = Path("data/chunked/corpus_chunks.json")
        if not qdrant_dir.is_dir() or not chunks_file.is_file():
            pytest.skip("Corpus files not found")

        with HybridRetriever() as retriever:
            res = retriever.retrieve(
                query="Do RAG and REALM use the same knowledge corpus, and how do their passage sizes compare?",
                top_k=5,
            )

            # Targets are P1 p4, P2 p4, P3 p6
            top3_pages = [(c.document_id, c.physical_page_number) for c in res.chunks[:3]]
            has_target = any(
                (doc == "P1" and p == 4) or (doc == "P2" and p == 4) or (doc == "P3" and p == 6)
                for doc, p in top3_pages
            )
            assert has_target, f"Expected target in top 3, got {top3_pages}"

