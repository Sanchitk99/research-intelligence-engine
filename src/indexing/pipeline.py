"""End-to-end indexing pipeline coordinating chunk loading, embedding, and vector storage.

Ensures strict schema validation, batching, collection compatibility, and error handling.
"""

import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from pydantic import ValidationError

from src.chunking.models import DocumentChunk
from src.indexing.models import (
    IndexingConfig,
    IndexingSummary,
    IndexingError,
    InvalidChunkRecordError,
)
from src.indexing.embeddings import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
)
from src.indexing.vector_store import QdrantStoreManager


class IndexingPipeline:
    """Coordinates batch embedding and Qdrant indexing for document chunks."""

    def __init__(
        self,
        config: IndexingConfig,
        embedder: Optional[BaseEmbedder] = None,
        store_manager: Optional[QdrantStoreManager] = None,
    ) -> None:
        """Initialize the pipeline with configuration and optional components.

        Args:
            config: Indexing configuration parameters.
            embedder: Optional embedder implementation (defaults to SentenceTransformerEmbedder).
            store_manager: Optional store manager (defaults to QdrantStoreManager on config.qdrant_path).
        """
        self.config = config
        self.embedder: BaseEmbedder = embedder or SentenceTransformerEmbedder(
            model_name=config.model_name,
            normalize_embeddings=config.normalize_embeddings,
        )
        self.store_manager: QdrantStoreManager = store_manager or QdrantStoreManager(
            qdrant_path=config.qdrant_path
        )
        self._owns_store_manager = store_manager is None

    def close(self) -> None:
        """Close managed store resources if owned."""
        if self._owns_store_manager:
            self.store_manager.close()

    def load_and_validate_chunks(
        self,
        chunks_file: Path,
    ) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """Load and validate chunk records from a JSON file.

        Args:
            chunks_file: Path to JSON file containing chunk records.

        Returns:
            Tuple of (valid_chunks, invalid_records).

        Raises:
            InvalidChunkRecordError: If file is missing or contains unparseable JSON.
        """
        path = Path(chunks_file)
        if not path.is_file():
            raise InvalidChunkRecordError(f"Chunks file does not exist: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidChunkRecordError(f"Failed to decode JSON from {path}: {e}") from e
        except Exception as e:
            raise InvalidChunkRecordError(f"Error reading {path}: {e}") from e

        # Support both JSON list of chunks and dictionary with 'chunks' key
        if isinstance(raw_data, list):
            items = raw_data
        elif isinstance(raw_data, dict) and "chunks" in raw_data and isinstance(raw_data["chunks"], list):
            items = raw_data["chunks"]
        else:
            raise InvalidChunkRecordError(
                f"Expected JSON array or object with 'chunks' array in {path}, got {type(raw_data).__name__}"
            )

        valid_chunks: List[DocumentChunk] = []
        invalid_records: List[Dict[str, Any]] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                invalid_records.append({"index": idx, "raw": item, "reason": "Item is not a dictionary"})
                continue

            try:
                chunk = DocumentChunk.model_validate(item)
                # Verify chunk text is non-empty
                if not chunk.text or not chunk.text.strip():
                    invalid_records.append({
                        "index": idx,
                        "chunk_id": getattr(chunk, "chunk_id", None),
                        "reason": "Chunk text is empty or whitespace-only",
                    })
                    continue
                valid_chunks.append(chunk)
            except ValidationError as ve:
                invalid_records.append({
                    "index": idx,
                    "chunk_id": item.get("chunk_id"),
                    "reason": f"Pydantic schema validation error: {ve}",
                })
            except Exception as e:
                invalid_records.append({
                    "index": idx,
                    "chunk_id": item.get("chunk_id"),
                    "reason": f"Unexpected error validating chunk: {e}",
                })

        return valid_chunks, invalid_records

    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        dry_run: bool = False,
    ) -> IndexingSummary:
        """Embed and upsert chunks in batches into Qdrant.

        Args:
            chunks: List of validated DocumentChunk records.
            dry_run: If True, validate model and shapes without writing to Qdrant.

        Returns:
            IndexingSummary with detailed execution statistics.
        """
        start_time = time.time()
        vector_dim = self.embedder.embedding_dim
        model_name = self.embedder.model_name
        collection_name = self.config.collection_name

        if not dry_run:
            # Ensure collection exists and matches model dimensionality
            self.store_manager.ensure_collection(
                collection_name=collection_name,
                expected_dim=vector_dim,
            )

        chunks_embedded = 0
        chunks_indexed = 0
        batch_size = self.config.batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c.text for c in batch]

            # Generate batch embeddings (verbatim texts unchanged)
            embeddings = self.embedder.embed_batch(batch_texts, batch_size=batch_size)
            chunks_embedded += len(embeddings)

            if not dry_run:
                upserted = self.store_manager.upsert_chunks(
                    collection_name=collection_name,
                    chunks=batch,
                    embeddings=embeddings,
                )
                chunks_indexed += upserted

        final_points = 0
        if not dry_run:
            final_points = self.store_manager.get_point_count(collection_name)

        elapsed = time.time() - start_time

        return IndexingSummary(
            chunks_read=len(chunks),
            chunks_embedded=chunks_embedded,
            chunks_indexed=chunks_indexed,
            chunks_skipped=0,
            chunks_failed=0,
            model_name=model_name,
            vector_dim=vector_dim,
            collection_name=collection_name,
            total_points_in_collection=final_points,
            dry_run=dry_run,
            execution_time_seconds=round(elapsed, 3),
        )

    def run(
        self,
        chunks_file: Path,
        dry_run: bool = False,
    ) -> IndexingSummary:
        """Full pipeline execution: read, validate, embed, and index chunks.

        Args:
            chunks_file: Path to chunks JSON artifact.
            dry_run: If True, validate without persisting points.

        Returns:
            IndexingSummary report.
        """
        try:
            valid_chunks, invalid_records = self.load_and_validate_chunks(chunks_file)
            summary = self.index_chunks(valid_chunks, dry_run=dry_run)
            summary.chunks_read = len(valid_chunks) + len(invalid_records)
            summary.chunks_failed = len(invalid_records)
            return summary
        finally:
            self.close()

