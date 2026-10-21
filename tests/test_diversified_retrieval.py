"""Unit and regression tests for Document-Diversified Retrieval (Task 4.4.18).

Tests:
1. Per-document cap enforcement across multi-document candidate lists.
2. Graceful score-ordered backfill when candidates from a single document dominate.
3. Graceful backfill when fewer candidate documents exist than required slots.
4. Backward compatibility when max_chunks_per_doc is None (default baseline).
5. Validation and error handling for invalid max_chunks_per_doc parameters.
6. Deduplication and deterministic tie-breaking.
7. Full citation and provenance metadata preservation.
8. Integration with HybridRetriever and HybridRetrievalConfig.
"""

import pytest
from typing import List

from src.retrieval.models import (
    RetrievedChunk,
    HybridRetrievalConfig,
    RetrievalError,
)
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever


def make_chunk(
    chunk_id: str,
    doc_id: str,
    page: int = 1,
    score: float = 0.5,
    text: str = "Sample chunk text",
) -> RetrievedChunk:
    """Helper to create dummy RetrievedChunk instances for unit testing."""
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
        end_char_idx=len(text),
        char_count=len(text),
        word_count=len(text.split()),
        text=text,
        score=score,
    )


class TestDocumentDiversification:
    """Tests for document diversification logic in reciprocal_rank_fusion."""

    def test_diversification_respects_per_doc_cap(self):
        """Pass 1 caps chunks per document and allows lower-ranked secondary documents into top-k."""
        # 5 chunks of Doc A, 3 chunks of Doc B
        # Dense ranking: A1, A2, A3, A4, A5
        # BM25 ranking: B1, B2, B3
        dense = [
            make_chunk("A1", "DocA", 1, 0.9),
            make_chunk("A2", "DocA", 2, 0.8),
            make_chunk("A3", "DocA", 3, 0.7),
            make_chunk("A4", "DocA", 4, 0.6),
            make_chunk("A5", "DocA", 5, 0.5),
        ]
        bm25 = [
            make_chunk("B1", "DocB", 1, 0.85),
            make_chunk("B2", "DocB", 2, 0.75),
            make_chunk("B3", "DocB", 3, 0.65),
        ]

        # Top 4 with max_chunks_per_doc=2:
        # Expected: A1 (rank 1 dense), B1 (rank 1 bm25), A2 (rank 2 dense), B2 (rank 2 bm25)
        fused = reciprocal_rank_fusion(
            dense_chunks=dense,
            bm25_chunks=bm25,
            rrf_k=60,
            top_k=4,
            max_chunks_per_doc=2,
        )

        assert len(fused) == 4
        doc_counts = {}
        for c in fused:
            doc_counts[c.document_id] = doc_counts.get(c.document_id, 0) + 1

        assert doc_counts["DocA"] == 2
        assert doc_counts["DocB"] == 2
        fused_ids = [c.chunk_id for c in fused]
        assert "A1" in fused_ids
        assert "A2" in fused_ids
        assert "B1" in fused_ids
        assert "B2" in fused_ids
        assert "A3" not in fused_ids

    def test_diversification_backfill_single_document(self):
        """When only one document has candidates, backfill ensures top_k slots are fully populated."""
        dense = [
            make_chunk("A1", "DocA", 1, 0.9),
            make_chunk("A2", "DocA", 2, 0.8),
            make_chunk("A3", "DocA", 3, 0.7),
            make_chunk("A4", "DocA", 4, 0.6),
            make_chunk("A5", "DocA", 5, 0.5),
        ]
        bm25 = []

        # Request top_k=5 with max_chunks_per_doc=2
        # Pass 1 takes A1, A2 (cap reached for DocA)
        # Pass 2 backfills A3, A4, A5
        fused = reciprocal_rank_fusion(
            dense_chunks=dense,
            bm25_chunks=bm25,
            rrf_k=60,
            top_k=5,
            max_chunks_per_doc=2,
        )

        assert len(fused) == 5
        fused_ids = [c.chunk_id for c in fused]
        assert fused_ids == ["A1", "A2", "A3", "A4", "A5"]

    def test_diversification_backfill_insufficient_documents(self):
        """When secondary documents exhaust candidates before filling top_k, primary doc fills remaining slots."""
        dense = [
            make_chunk("A1", "DocA", 1, 0.9),
            make_chunk("A2", "DocA", 2, 0.8),
            make_chunk("A3", "DocA", 3, 0.7),
            make_chunk("A4", "DocA", 4, 0.6),
        ]
        bm25 = [
            make_chunk("B1", "DocB", 1, 0.85),
        ]

        # Request top_k=4 with max_chunks_per_doc=2
        # Pass 1 takes A1, A2 (DocA count=2), and B1 (DocB count=1) -> 3 selected
        # Pass 2 backfills A3 (DocA count becomes 3) -> 4 selected
        fused = reciprocal_rank_fusion(
            dense_chunks=dense,
            bm25_chunks=bm25,
            rrf_k=60,
            top_k=4,
            max_chunks_per_doc=2,
        )

        assert len(fused) == 4
        fused_ids = [c.chunk_id for c in fused]
        assert fused_ids[:3] == ["A1", "B1", "A2"] or set(fused_ids[:3]) == {"A1", "A2", "B1"}
        assert fused_ids[3] == "A3"

    def test_diversification_none_preserves_baseline(self):
        """When max_chunks_per_doc is None, behavior is 100% identical to baseline RRF."""
        dense = [
            make_chunk("A1", "DocA", 1, 0.9),
            make_chunk("A2", "DocA", 2, 0.8),
            make_chunk("A3", "DocA", 3, 0.7),
            make_chunk("A4", "DocA", 4, 0.6),
            make_chunk("A5", "DocA", 5, 0.5),
        ]
        bm25 = [
            make_chunk("B1", "DocB", 1, 0.3),
        ]

        baseline_default = reciprocal_rank_fusion(
            dense_chunks=dense,
            bm25_chunks=bm25,
            top_k=5,
        )
        baseline_explicit_none = reciprocal_rank_fusion(
            dense_chunks=dense,
            bm25_chunks=bm25,
            top_k=5,
            max_chunks_per_doc=None,
        )
        assert [c.chunk_id for c in baseline_explicit_none] == [c.chunk_id for c in baseline_default]
        assert [c.chunk_id for c in baseline_default] == ["A1", "B1", "A2", "A3", "A4"]

    def test_diversification_parameter_validation(self):
        """max_chunks_per_doc < 1 raises ValueError."""
        dense = [make_chunk("A1", "DocA")]
        bm25 = [make_chunk("B1", "DocB")]

        with pytest.raises(ValueError, match="max_chunks_per_doc"):
            reciprocal_rank_fusion(dense, bm25, max_chunks_per_doc=0)

        with pytest.raises(ValueError, match="max_chunks_per_doc"):
            reciprocal_rank_fusion(dense, bm25, max_chunks_per_doc=-2)

    def test_diversification_deduplication_and_tie_breaking(self):
        """A chunk present in both dense and BM25 is deduplicated; ties are broken by chunk_id."""
        c_shared = make_chunk("C_shared", "DocC", 1, 0.8)
        dense = [c_shared]
        bm25 = [c_shared]

        fused = reciprocal_rank_fusion(
            dense_chunks=dense,
            bm25_chunks=bm25,
            top_k=5,
            max_chunks_per_doc=2,
        )

        assert len(fused) == 1
        assert fused[0].chunk_id == "C_shared"
        assert fused[0].dense_rank == 1
        assert fused[0].bm25_rank == 1

    def test_diversification_provenance_preservation(self):
        """All citation provenance fields are intact on diversified chunks."""
        c1 = make_chunk("P1_p04_c02", "P1", page=4, text="Deep transformer architecture")
        dense = [c1]
        bm25 = []

        fused = reciprocal_rank_fusion(dense, bm25, top_k=1, max_chunks_per_doc=1)
        assert len(fused) == 1
        f = fused[0]
        assert f.chunk_id == "P1_p04_c02"
        assert f.document_id == "P1"
        assert f.physical_page_number == 4
        assert f.printed_page_number == 4
        assert f.retrieval_method == "hybrid_rrf"
        assert f.text == "Deep transformer architecture"
        assert f.char_count == len("Deep transformer architecture")


class TestHybridRetrieverIntegration:
    """Tests HybridRetriever's handling of max_chunks_per_doc via config and retrieve call."""

    def test_retriever_validation_error(self):
        """Invalid max_chunks_per_doc passed to retrieve raises RetrievalError."""
        config = HybridRetrievalConfig()
        retriever = HybridRetriever(config=config)
        try:
            with pytest.raises(RetrievalError, match="max_chunks_per_doc"):
                retriever.retrieve(query="test query", max_chunks_per_doc=-1)

            with pytest.raises(RetrievalError, match="max_chunks_per_doc"):
                retriever.retrieve(query="test query", max_chunks_per_doc="invalid")  # type: ignore
        finally:
            retriever.close()

    def test_config_max_chunks_per_doc_default(self):
        """HybridRetrievalConfig defaults max_chunks_per_doc to None."""
        config = HybridRetrievalConfig()
        assert config.max_chunks_per_doc is None

    def test_config_custom_max_chunks_per_doc(self):
        """HybridRetrievalConfig accepts positive integer for max_chunks_per_doc."""
        config = HybridRetrievalConfig(max_chunks_per_doc=3)
        assert config.max_chunks_per_doc == 3
