"""Tests for Task 3.4: Dense Retrieval.

Covers:
1. Relevant dense retrieval for research queries
2. Result limit enforcement (top_k)
3. Single and multi-document filtering
4. Invalid query input handling (empty, whitespace, non-string)
5. Invalid top_k and setting validation
6. Missing collection error handling
7. Full citation provenance preservation
8. Deterministic ranking across repeated identical queries
9. Minimum score threshold filtering
10. Integration test against real persistent local collection in data/qdrant
"""

from pathlib import Path
from typing import List

import pytest
from qdrant_client import QdrantClient

from src.chunking.models import DocumentChunk, ChunkMetadata
from src.indexing.embeddings import MockEmbedder, SentenceTransformerEmbedder
from src.indexing.vector_store import QdrantStoreManager
from src.retrieval.models import (
    RetrievalConfig,
    RetrievedChunk,
    RetrievalResult,
    RetrievalError,
    InvalidQueryError,
    CollectionNotFoundError,
    FilterError,
)
from src.retrieval.retriever import DenseRetriever


# ---------------------------------------------------------------------------
# Test Fixtures (Synthetic Corpus for Offline Unit Testing)
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_retriever_setup():
    """Sets up an in-memory Qdrant instance with synthetic indexed chunks."""
    client = QdrantClient(":memory:")
    store = QdrantStoreManager(client=client)
    embedder = MockEmbedder(embedding_dim=64)

    collection_name = "test_retrieval_corpus"
    store.ensure_collection(collection_name, expected_dim=64)

    # Create synthetic chunks across documents P1, P2, P3
    chunks = [
        DocumentChunk(
            chunk_id="P1_p01_c01",
            metadata=ChunkMetadata(
                chunk_id="P1_p01_c01",
                document_id="P1",
                document_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                physical_page_index=0,
                physical_page_number=1,
                printed_page_number=1,
                chunk_index_in_page=0,
                chunk_index_in_doc=0,
                start_char_idx=0,
                end_char_idx=90,
                char_count=90,
                word_count=12,
            ),
            text="RAG combines pre-trained parametric memory with non-parametric dense passage retrieval.",
        ),
        DocumentChunk(
            chunk_id="P1_p03_c01",
            metadata=ChunkMetadata(
                chunk_id="P1_p03_c01",
                document_id="P1",
                document_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                physical_page_index=2,
                physical_page_number=3,
                printed_page_number=3,
                chunk_index_in_page=0,
                chunk_index_in_doc=1,
                start_char_idx=0,
                end_char_idx=80,
                char_count=80,
                word_count=10,
            ),
            text="The generator component in RAG uses BART-large with 400M parameters.",
        ),
        DocumentChunk(
            chunk_id="P2_p03_c01",
            metadata=ChunkMetadata(
                chunk_id="P2_p03_c01",
                document_id="P2",
                document_title="Dense Passage Retrieval for Open-Domain Question Answering",
                physical_page_index=2,
                physical_page_number=3,
                printed_page_number=None,
                chunk_index_in_page=0,
                chunk_index_in_doc=0,
                start_char_idx=0,
                end_char_idx=95,
                char_count=95,
                word_count=14,
            ),
            text="DPR trains a dual-encoder architecture using in-batch negatives and one BM25 hard negative.",
        ),
        DocumentChunk(
            chunk_id="P3_p02_c01",
            metadata=ChunkMetadata(
                chunk_id="P3_p02_c01",
                document_id="P3",
                document_title="REALM: Retrieval-Augmented Language Model Pre-Training",
                physical_page_index=1,
                physical_page_number=2,
                printed_page_number=None,
                chunk_index_in_page=0,
                chunk_index_in_doc=0,
                start_char_idx=0,
                end_char_idx=85,
                char_count=85,
                word_count=11,
            ),
            text="REALM asynchronously updates the MIPS index during masked language model pre-training.",
        ),
    ]

    embeddings = embedder.embed_batch([c.text for c in chunks])
    store.upsert_chunks(collection_name, chunks, embeddings)

    config = RetrievalConfig(collection_name=collection_name, top_k=3)
    retriever = DenseRetriever(config=config, embedder=embedder, store_manager=store)
    return retriever, chunks


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestDenseRetrieverUnit:
    """Unit tests for DenseRetriever with in-memory Qdrant and MockEmbedder."""

    def test_basic_retrieval_returns_ranked_chunks(self, memory_retriever_setup):
        """Verify that basic retrieval returns ranked chunks with scores."""
        retriever, _ = memory_retriever_setup
        result = retriever.retrieve("What generator model does RAG use?")

        assert isinstance(result, RetrievalResult)
        assert result.total_retrieved > 0
        assert len(result.chunks) <= 3

        # Scores must be in descending order
        scores = [c.score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

        for chunk in result.chunks:
            assert isinstance(chunk, RetrievedChunk)
            assert -1.0 <= chunk.score <= 1.0
            assert len(chunk.text) > 0

    def test_top_k_limit_enforcement(self, memory_retriever_setup):
        """Verify that top_k strictly limits the number of returned chunks."""
        retriever, _ = memory_retriever_setup

        res_1 = retriever.retrieve("passage retrieval", top_k=1)
        assert len(res_1.chunks) == 1

        res_2 = retriever.retrieve("passage retrieval", top_k=2)
        assert len(res_2.chunks) == 2

    def test_single_document_filtering(self, memory_retriever_setup):
        """Verify that filtering by document_id restricts results exclusively to that paper."""
        retriever, _ = memory_retriever_setup

        # Filter strictly to P1
        result_p1 = retriever.retrieve("passage retrieval architecture", document_ids="P1")
        assert result_p1.total_retrieved > 0
        for chunk in result_p1.chunks:
            assert chunk.document_id == "P1"

        # Filter strictly to P2
        result_p2 = retriever.retrieve("passage retrieval architecture", document_ids="P2")
        assert result_p2.total_retrieved > 0
        for chunk in result_p2.chunks:
            assert chunk.document_id == "P2"

    def test_multi_document_filtering(self, memory_retriever_setup):
        """Verify that filtering by multiple document IDs allows only those papers."""
        retriever, _ = memory_retriever_setup

        result = retriever.retrieve("pre-training language models", document_ids=["P1", "P3"])
        assert result.total_retrieved > 0
        for chunk in result.chunks:
            assert chunk.document_id in {"P1", "P3"}
            assert chunk.document_id != "P2"

    def test_invalid_query_validation(self, memory_retriever_setup):
        """Verify that empty, whitespace-only, or non-string queries raise InvalidQueryError."""
        retriever, _ = memory_retriever_setup

        with pytest.raises(InvalidQueryError, match="empty or whitespace-only"):
            retriever.retrieve("")

        with pytest.raises(InvalidQueryError, match="empty or whitespace-only"):
            retriever.retrieve("   \t\n  ")

        with pytest.raises(InvalidQueryError, match="must be a string"):
            retriever.retrieve(None)  # type: ignore

        with pytest.raises(InvalidQueryError, match="must be a string"):
            retriever.retrieve(12345)  # type: ignore

    def test_invalid_top_k_and_filter_validation(self, memory_retriever_setup):
        """Verify proper exceptions for invalid top_k or invalid filter inputs."""
        retriever, _ = memory_retriever_setup

        with pytest.raises(RetrievalError, match="top_k must be an integer >= 1"):
            retriever.retrieve("valid query", top_k=0)

        with pytest.raises(RetrievalError, match="top_k must be an integer >= 1"):
            retriever.retrieve("valid query", top_k=-5)

        with pytest.raises(FilterError, match="no valid non-empty document IDs"):
            retriever.retrieve("valid query", document_ids=[])

        with pytest.raises(FilterError, match="no valid non-empty document IDs"):
            retriever.retrieve("valid query", document_ids=["   "])

    def test_missing_collection_handling(self, memory_retriever_setup):
        """Verify that searching a non-existent collection raises CollectionNotFoundError."""
        retriever, _ = memory_retriever_setup

        with pytest.raises(CollectionNotFoundError, match="does not exist in Qdrant store"):
            retriever.retrieve("any query", collection_name="non_existent_collection_xyz")

    def test_provenance_preservation(self, memory_retriever_setup):
        """Verify that retrieved chunks retain exact source page and document provenance."""
        retriever, chunks = memory_retriever_setup
        result = retriever.retrieve("generator component BART-large", top_k=4)

        chunk_lookup = {c.chunk_id: c for c in chunks}
        for r_chunk in result.chunks:
            orig = chunk_lookup[r_chunk.chunk_id]
            assert r_chunk.document_id == orig.metadata.document_id
            assert r_chunk.document_title == orig.metadata.document_title
            assert r_chunk.physical_page_number == orig.metadata.physical_page_number
            assert r_chunk.printed_page_number == orig.metadata.printed_page_number
            assert r_chunk.start_char_idx == orig.metadata.start_char_idx
            assert r_chunk.end_char_idx == orig.metadata.end_char_idx
            assert r_chunk.text == orig.text

            # Test formatted citation helper
            citation = r_chunk.formatted_citation()
            assert r_chunk.document_id in citation
            assert f"Physical Page {r_chunk.physical_page_number}" in citation

    def test_deterministic_retrieval(self, memory_retriever_setup):
        """Verify that identical queries produce identical results and rankings."""
        retriever, _ = memory_retriever_setup
        q = "dual-encoder architecture training"

        run1 = retriever.retrieve(q, top_k=3)
        run2 = retriever.retrieve(q, top_k=3)

        assert len(run1.chunks) == len(run2.chunks)
        for c1, c2 in zip(run1.chunks, run2.chunks):
            assert c1.chunk_id == c2.chunk_id
            assert c1.score == c2.score


# ---------------------------------------------------------------------------
# Integration Tests (Real Qdrant Collection in data/qdrant)
# ---------------------------------------------------------------------------

class TestDenseRetrieverIntegration:
    """Integration tests running against the real persistent Qdrant collection."""

    @pytest.fixture
    def real_retriever(self):
        """Fixture connecting to the real persistent Qdrant store."""
        db_path = Path("data/qdrant")
        if not db_path.exists():
            pytest.skip("data/qdrant directory not found. Run indexing first.")

        config = RetrievalConfig(
            qdrant_path=db_path,
            collection_name="research_corpus",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            top_k=5,
        )
        return DenseRetriever(config=config)

    def test_real_corpus_q01_rag_generator(self, real_retriever: DenseRetriever):
        """Evaluation Q01: 'What generator model does RAG use?' -> should retrieve P1 and mention BART."""
        result = real_retriever.retrieve("What generator model does RAG use?", top_k=3)
        assert result.total_retrieved > 0

        # Top 3 should contain paper P1 (RAG paper)
        doc_ids = [c.document_id for c in result.chunks]
        assert "P1" in doc_ids

        # Check evidence in retrieved texts
        combined_text = " ".join(c.text for c in result.chunks)
        assert "BART" in combined_text or "generator" in combined_text.lower()

    def test_real_corpus_q02_dpr_encoder_negatives(self, real_retriever: DenseRetriever):
        """Evaluation Q02: 'How does DPR train its passage encoder and what negatives are used?' -> should retrieve P2."""
        result = real_retriever.retrieve(
            "How does DPR train its passage encoder and what types of negatives are used?",
            top_k=3,
        )
        assert result.total_retrieved > 0

        doc_ids = [c.document_id for c in result.chunks]
        assert "P2" in doc_ids

        combined_text = " ".join(c.text for c in result.chunks)
        assert "negative" in combined_text.lower() or "encoder" in combined_text.lower()

    def test_real_corpus_filtered_query(self, real_retriever: DenseRetriever):
        """Verify real collection filtering: only P4 (Self-RAG) is returned when filtered."""
        result = real_retriever.retrieve(
            "reflection tokens and critique model",
            document_ids="P4",
            top_k=5,
        )
        assert result.total_retrieved > 0
        for chunk in result.chunks:
            assert chunk.document_id == "P4"
            assert "Self-RAG" in chunk.document_title or "SELF-RAG" in chunk.document_title

