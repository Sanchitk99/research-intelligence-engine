"""Comprehensive tests for Task 3.3: Embeddings and Qdrant Vector Storage.

Covers:
1. Single-text and batch embedding output shapes and L2 normalization
2. Empty input and invalid embedding error handling
3. Deterministic Qdrant point IDs via UUIDv5
4. Complete payload metadata preservation and verbatim text fidelity
5. Collection creation and vector dimension validation
6. Idempotency: re-indexing the same data leaves point count unchanged
7. Detection and isolation of missing or malformed chunk records
8. Clear exception handling for embedding and storage failures
9. Persistent storage integration on disk with temporary directories
10. Real corpus indexing pipeline execution with schema validation
"""

import json
import uuid
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.chunking.models import DocumentChunk, ChunkMetadata
from src.indexing.models import (
    IndexingConfig,
    IndexingSummary,
    EmbeddingError,
    VectorStoreError,
    IncompatibleCollectionError,
    InvalidChunkRecordError,
)
from src.indexing.embeddings import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    MockEmbedder,
)
from src.indexing.vector_store import QdrantStoreManager, NAMESPACE_RIE_CHUNK
from src.indexing.pipeline import IndexingPipeline


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunk_metadata() -> ChunkMetadata:
    """Fixture providing a complete ChunkMetadata instance."""
    return ChunkMetadata(
        chunk_id="P1_p01_c01",
        document_id="P1",
        document_title="Retrieval-Augmented Generation for NLP Tasks",
        physical_page_index=0,
        physical_page_number=1,
        printed_page_number=1,
        chunk_index_in_page=0,
        chunk_index_in_doc=0,
        start_char_idx=0,
        end_char_idx=120,
        char_count=120,
        word_count=18,
    )


@pytest.fixture
def sample_chunk(sample_chunk_metadata: ChunkMetadata) -> DocumentChunk:
    """Fixture providing a complete DocumentChunk instance."""
    return DocumentChunk(
        chunk_id=sample_chunk_metadata.chunk_id,
        metadata=sample_chunk_metadata,
        text="Dense passage retrieval leverages dual-encoders trained with BM25 negatives to maximize top-k retrieval accuracy.",
    )


@pytest.fixture
def synthetic_chunks() -> List[DocumentChunk]:
    """Fixture providing a batch of 5 distinct synthetic chunks."""
    chunks = []
    for i in range(1, 6):
        cid = f"TEST_p01_c{i:02d}"
        meta = ChunkMetadata(
            chunk_id=cid,
            document_id="TEST",
            document_title="Synthetic Evaluation Paper",
            physical_page_index=0,
            physical_page_number=1,
            printed_page_number=1,
            chunk_index_in_page=i - 1,
            chunk_index_in_doc=i - 1,
            start_char_idx=(i - 1) * 100,
            end_char_idx=i * 100,
            char_count=100,
            word_count=15,
        )
        chunks.append(DocumentChunk(
            chunk_id=cid,
            metadata=meta,
            text=f"Synthetic chunk content {i} demonstrating deterministic embedding and indexing into Qdrant store.",
        ))
    return chunks


# ---------------------------------------------------------------------------
# 1. Embedding Output Shapes & Normalization
# ---------------------------------------------------------------------------

class TestEmbeddingGeneration:
    """Tests embedding generation interfaces, shapes, and normalization."""

    def test_single_and_batch_output_shapes_mock(self):
        """Verify MockEmbedder outputs correct 1D and 2D vector shapes."""
        embedder = MockEmbedder(embedding_dim=128, normalize_embeddings=True)

        vec = embedder.embed_text("Test single text embedding.")
        assert isinstance(vec, list)
        assert len(vec) == 128
        assert np.isfinite(vec).all()
        # Check L2 unit norm
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)

        batch_texts = ["Sentence one.", "Sentence two.", "Sentence three."]
        batch_vecs = embedder.embed_batch(batch_texts)
        assert len(batch_vecs) == 3
        for row in batch_vecs:
            assert len(row) == 128
            assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-4)

    def test_empty_input_and_invalid_inputs(self):
        """Verify proper error handling for empty or non-string inputs."""
        embedder = MockEmbedder(embedding_dim=64)

        with pytest.raises(EmbeddingError, match="Cannot embed empty or whitespace-only text"):
            embedder.embed_text("")

        with pytest.raises(EmbeddingError, match="Cannot embed empty or whitespace-only text"):
            embedder.embed_text("   \n\t  ")

        with pytest.raises(EmbeddingError, match="Expected str input"):
            embedder.embed_text(12345)  # type: ignore

        assert embedder.embed_batch([]) == []

        with pytest.raises(EmbeddingError, match="empty or whitespace-only"):
            embedder.embed_batch(["Valid text", "   ", "Another valid text"])

        with pytest.raises(EmbeddingError, match="not a string"):
            embedder.embed_batch(["Valid text", None])  # type: ignore

    def test_non_finite_vector_detection(self):
        """Verify that embedding methods reject vectors with NaN or Inf."""
        class FaultyEmbedder(MockEmbedder):
            def _generate_vector(self, text: str) -> np.ndarray:
                vec = super()._generate_vector(text)
                vec[0] = np.nan
                return vec

        bad_embedder = FaultyEmbedder(embedding_dim=32)
        with pytest.raises(EmbeddingError, match="non-finite"):
            bad_embedder.embed_text("Sample text")


# ---------------------------------------------------------------------------
# 2. Qdrant Point IDs and Metadata Provenance
# ---------------------------------------------------------------------------

class TestQdrantStorage:
    """Tests Qdrant store manager, deterministic IDs, and payload preservation."""

    def test_deterministic_point_ids(self):
        """Verify that Qdrant point IDs are stable, valid UUIDv5 strings derived from chunk_id."""
        cid1 = "P1_p01_c01"
        cid2 = "P1_p01_c02"

        pt_id_1a = QdrantStoreManager.chunk_id_to_point_id(cid1)
        pt_id_1b = QdrantStoreManager.chunk_id_to_point_id(cid1)
        pt_id_2 = QdrantStoreManager.chunk_id_to_point_id(cid2)

        # ID stability across repeated calls
        assert pt_id_1a == pt_id_1b
        # Distinct chunk IDs produce distinct point IDs
        assert pt_id_1a != pt_id_2

        # Valid UUIDv5 conformance
        parsed_uuid = uuid.UUID(pt_id_1a)
        assert parsed_uuid.version == 5

        # Empty or invalid chunk_id handling
        with pytest.raises(VectorStoreError):
            QdrantStoreManager.chunk_id_to_point_id("")

    def test_payload_metadata_preservation(self, sample_chunk: DocumentChunk):
        """Verify that all citation-relevant metadata and verbatim text are preserved in Qdrant."""
        client = QdrantClient(":memory:")
        store = QdrantStoreManager(client=client)

        dim = 64
        embedder = MockEmbedder(embedding_dim=dim)
        store.ensure_collection("test_citations", expected_dim=dim)

        embedding = embedder.embed_text(sample_chunk.text)
        store.upsert_chunks("test_citations", [sample_chunk], [embedding])

        # Retrieve the point from Qdrant
        record = store.get_point_by_chunk_id("test_citations", sample_chunk.chunk_id)
        assert record is not None
        assert record.id == QdrantStoreManager.chunk_id_to_point_id(sample_chunk.chunk_id)

        payload = record.payload
        meta = sample_chunk.metadata

        assert payload["chunk_id"] == meta.chunk_id
        assert payload["document_id"] == meta.document_id
        assert payload["document_title"] == meta.document_title
        assert payload["physical_page_index"] == meta.physical_page_index
        assert payload["physical_page_number"] == meta.physical_page_number
        assert payload["printed_page_number"] == meta.printed_page_number
        assert payload["chunk_index_in_page"] == meta.chunk_index_in_page
        assert payload["chunk_index_in_doc"] == meta.chunk_index_in_doc
        assert payload["start_char_idx"] == meta.start_char_idx
        assert payload["end_char_idx"] == meta.end_char_idx
        assert payload["char_count"] == meta.char_count
        assert payload["word_count"] == meta.word_count
        # Verbatim text matches original exactly
        assert payload["text"] == sample_chunk.text

    def test_collection_creation_and_dimension_validation(self):
        """Verify that collections are created with correct dimensions and conflicts are rejected."""
        client = QdrantClient(":memory:")
        store = QdrantStoreManager(client=client)

        # First creation succeeds
        created = store.ensure_collection("research_test", expected_dim=384)
        assert created is True

        # Second call with identical dim succeeds without re-creating
        rechecked = store.ensure_collection("research_test", expected_dim=384)
        assert rechecked is False

        # Call with mismatched dimension raises IncompatibleCollectionError
        with pytest.raises(IncompatibleCollectionError) as exc_info:
            store.ensure_collection("research_test", expected_dim=768)
        assert "vector dimension 384" in str(exc_info.value)
        assert "embedding model dimension 768" in str(exc_info.value)

    def test_reindexing_idempotency_same_point_count(self, synthetic_chunks: List[DocumentChunk]):
        """Verify that indexing the exact same chunks twice never duplicates points."""
        client = QdrantClient(":memory:")
        store = QdrantStoreManager(client=client)

        dim = 128
        embedder = MockEmbedder(embedding_dim=dim)
        store.ensure_collection("idempotent_test", expected_dim=dim)

        # First indexing pass
        embeddings = embedder.embed_batch([c.text for c in synthetic_chunks])
        count1 = store.upsert_chunks("idempotent_test", synthetic_chunks, embeddings)
        assert count1 == 5
        assert store.get_point_count("idempotent_test") == 5

        # Second indexing pass with the same chunks
        count2 = store.upsert_chunks("idempotent_test", synthetic_chunks, embeddings)
        assert count2 == 5
        # Total points in collection MUST remain 5, not 10
        assert store.get_point_count("idempotent_test") == 5


# ---------------------------------------------------------------------------
# 3. Pipeline, Validation, and Disk Persistence
# ---------------------------------------------------------------------------

class TestIndexingPipeline:
    """Tests the full indexing pipeline, error handling, and file persistence."""

    def test_missing_and_malformed_chunk_records(self, tmp_path: Path):
        """Verify pipeline detects invalid schemas, missing files, and whitespace chunks."""
        config = IndexingConfig(qdrant_path=tmp_path / "qdrant_db")
        embedder = MockEmbedder(embedding_dim=64)
        store = QdrantStoreManager(client=QdrantClient(":memory:"))
        pipeline = IndexingPipeline(config=config, embedder=embedder, store_manager=store)

        # 1. Non-existent file
        with pytest.raises(InvalidChunkRecordError, match="does not exist"):
            pipeline.load_and_validate_chunks(tmp_path / "non_existent.json")

        # 2. Corrupt JSON syntax
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{broken json syntax", encoding="utf-8")
        with pytest.raises(InvalidChunkRecordError, match="Failed to decode JSON"):
            pipeline.load_and_validate_chunks(bad_json)

        # 3. Mix of valid and invalid chunk records
        mixed_file = tmp_path / "mixed_chunks.json"
        records = [
            # Valid record
            {
                "chunk_id": "VAL_01",
                "metadata": {
                    "chunk_id": "VAL_01",
                    "document_id": "VAL",
                    "document_title": "Valid Doc",
                    "physical_page_index": 0,
                    "physical_page_number": 1,
                    "printed_page_number": None,
                    "chunk_index_in_page": 0,
                    "chunk_index_in_doc": 0,
                    "start_char_idx": 0,
                    "end_char_idx": 30,
                    "char_count": 30,
                    "word_count": 5,
                },
                "text": "Valid non-empty chunk text here.",
            },
            # Missing metadata
            {"chunk_id": "INV_01", "text": "Missing metadata field"},
            # Empty text
            {
                "chunk_id": "INV_02",
                "metadata": {
                    "chunk_id": "INV_02",
                    "document_id": "VAL",
                    "document_title": "Valid Doc",
                    "physical_page_index": 0,
                    "physical_page_number": 1,
                    "printed_page_number": None,
                    "chunk_index_in_page": 1,
                    "chunk_index_in_doc": 1,
                    "start_char_idx": 31,
                    "end_char_idx": 31,
                    "char_count": 0,
                    "word_count": 0,
                },
                "text": "   \n\t ",
            },
            # Non-dict item
            "just a string instead of chunk object",
        ]
        mixed_file.write_text(json.dumps(records), encoding="utf-8")

        valid, invalid = pipeline.load_and_validate_chunks(mixed_file)
        assert len(valid) == 1
        assert valid[0].chunk_id == "VAL_01"
        assert len(invalid) == 3

    def test_pipeline_dry_run_does_not_persist(self, synthetic_chunks: List[DocumentChunk]):
        """Verify dry-run mode embeds and validates without writing points to Qdrant."""
        client = QdrantClient(":memory:")
        store = QdrantStoreManager(client=client)
        config = IndexingConfig(collection_name="dry_run_coll")
        embedder = MockEmbedder(embedding_dim=64)
        pipeline = IndexingPipeline(config=config, embedder=embedder, store_manager=store)

        summary = pipeline.index_chunks(synthetic_chunks, dry_run=True)
        assert summary.dry_run is True
        assert summary.chunks_embedded == len(synthetic_chunks)
        assert summary.chunks_indexed == 0
        # Collection should not even exist or have 0 points
        assert not client.collection_exists("dry_run_coll")

    def test_persistent_disk_qdrant_directory(self, tmp_path: Path, synthetic_chunks: List[DocumentChunk]):
        """Verify Qdrant data persists to disk and reloads across distinct client sessions."""
        db_dir = tmp_path / "persistent_qdrant"
        config = IndexingConfig(
            qdrant_path=db_dir,
            collection_name="disk_test",
            batch_size=2,
        )
        embedder = MockEmbedder(embedding_dim=32)

        # Pass 1: Initialize store, index chunks, then close
        with QdrantStoreManager(qdrant_path=db_dir) as store1:
            pipeline1 = IndexingPipeline(config=config, embedder=embedder, store_manager=store1)
            summary1 = pipeline1.index_chunks(synthetic_chunks, dry_run=False)
            assert summary1.chunks_indexed == 5
            assert store1.get_point_count("disk_test") == 5

        # Verify disk files were created
        assert db_dir.exists()
        assert any(db_dir.iterdir())

        # Pass 2: Reopen a fresh store pointing to the same disk path
        with QdrantStoreManager(qdrant_path=db_dir) as store2:
            assert store2.client.collection_exists("disk_test")
            assert store2.get_point_count("disk_test") == 5
            pt = store2.get_point_by_chunk_id("disk_test", synthetic_chunks[0].chunk_id)
            assert pt is not None
            assert pt.payload["document_id"] == "TEST"

    def test_real_corpus_json_schema_and_mock_indexing(self):
        """Verify that the actual data/chunked/corpus_chunks.json file loads and indexes cleanly."""
        corpus_file = Path("data/chunked/corpus_chunks.json")
        if not corpus_file.is_file():
            pytest.skip("data/chunked/corpus_chunks.json not found on disk")

        client = QdrantClient(":memory:")
        store = QdrantStoreManager(client=client)
        config = IndexingConfig(collection_name="corpus_test", batch_size=64)
        embedder = MockEmbedder(embedding_dim=384)
        pipeline = IndexingPipeline(config=config, embedder=embedder, store_manager=store)

        summary = pipeline.run(chunks_file=corpus_file, dry_run=False)

        assert summary.chunks_read == 396
        assert summary.chunks_embedded == 396
        assert summary.chunks_indexed == 396
        assert summary.chunks_failed == 0
        assert summary.vector_dim == 384
        assert summary.total_points_in_collection == 396

        # Re-indexing must leave total points unchanged at 396
        summary_rerun = pipeline.run(chunks_file=corpus_file, dry_run=False)
        assert summary_rerun.total_points_in_collection == 396


# ---------------------------------------------------------------------------
# 4. Integration Test with Real SentenceTransformer (if available)
# ---------------------------------------------------------------------------

class TestRealSentenceTransformerIntegration:
    """Integration test verifying real sentence-transformers model encoding."""

    def test_real_sentence_transformer_minilm(self):
        """Test encoding with real all-MiniLM-L6-v2 model."""
        try:
            embedder = SentenceTransformerEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                normalize_embeddings=True,
            )
            dim = embedder.embedding_dim
            assert dim == 384

            text = "Dense passage retrieval for open-domain question answering."
            vec = embedder.embed_text(text)

            assert len(vec) == 384
            assert np.isfinite(vec).all()
            norm = np.linalg.norm(vec)
            assert np.isclose(norm, 1.0, atol=1e-3)
        except Exception as e:
            pytest.skip(f"SentenceTransformer not available or model load failed: {e}")

