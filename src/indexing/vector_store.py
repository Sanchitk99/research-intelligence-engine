"""Vector storage manager interfacing with local Qdrant.

Handles collection lifecycle, deterministic point ID generation, validation,
and batch point upserts with full provenance metadata.
"""

import uuid
import warnings
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
    Record,
)

from src.chunking.models import DocumentChunk
from src.indexing.models import (
    VectorStoreError,
    IncompatibleCollectionError,
)

# Fixed deterministic namespace UUID for RIE chunks (RFC 4122 v5)
NAMESPACE_RIE_CHUNK = uuid.UUID("3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f")


class QdrantStoreManager:
    """Manages Qdrant client connection, collections, and point upserts."""

    def __init__(
        self,
        qdrant_path: Optional[Union[str, Path]] = None,
        client: Optional[QdrantClient] = None,
    ) -> None:
        """Initialize Qdrant store manager.

        Args:
            qdrant_path: Path to persistent local directory on disk.
            client: Optional pre-configured QdrantClient (e.g. for testing with :memory:).
        """
        if client is not None:
            self._client = client
            self._path = None
        elif qdrant_path is not None:
            self._path = Path(qdrant_path)
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
        else:
            self._path = None
            self._client = QdrantClient(":memory:")

    @property
    def client(self) -> QdrantClient:
        """Get underlying QdrantClient instance."""
        return self._client

    @property
    def storage_path(self) -> Optional[Path]:
        """Path to persistent storage directory if using disk storage."""
        return self._path

    def close(self) -> None:
        """Safely close client connections and flush lockfiles."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    def __enter__(self) -> "QdrantStoreManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def chunk_id_to_point_id(chunk_id: str) -> str:
        """Derive a stable, collision-free UUIDv5 point ID from a chunk ID string.

        Args:
            chunk_id: The unique chunk identifier (e.g. 'P1_p01_c01').

        Returns:
            Deterministic UUID string compatible with Qdrant point IDs.
        """
        if not chunk_id or not isinstance(chunk_id, str):
            raise VectorStoreError(f"Invalid chunk_id for point ID generation: {chunk_id!r}")
        return str(uuid.uuid5(NAMESPACE_RIE_CHUNK, chunk_id.strip()))

    def ensure_collection(
        self,
        collection_name: str,
        expected_dim: int,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Ensure a Qdrant collection exists and has compatible vector configuration.

        Args:
            collection_name: Name of target collection.
            expected_dim: Required vector dimensionality from embedding model.
            distance: Vector similarity metric (default Cosine).

        Returns:
            True if collection was freshly created; False if it already existed and is valid.

        Raises:
            IncompatibleCollectionError: If existing collection has conflicting dimension or config.
            VectorStoreError: If collection inspection or creation fails.
        """
        try:
            exists = self._client.collection_exists(collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to check existence of collection '{collection_name}': {e}") from e

        if exists:
            try:
                info = self._client.get_collection(collection_name)
            except Exception as e:
                raise VectorStoreError(f"Failed to inspect collection '{collection_name}': {e}") from e

            # Validate vector dimensions
            vectors_cfg = info.config.params.vectors
            existing_dim: Optional[int] = None
            if hasattr(vectors_cfg, "size"):
                existing_dim = vectors_cfg.size
            elif isinstance(vectors_cfg, dict):
                # Named vectors: inspect default or first
                first_cfg = next(iter(vectors_cfg.values()))
                if hasattr(first_cfg, "size"):
                    existing_dim = first_cfg.size

            if existing_dim is not None and existing_dim != expected_dim:
                raise IncompatibleCollectionError(
                    f"Collection '{collection_name}' already exists with vector dimension {existing_dim}, "
                    f"which does not match the embedding model dimension {expected_dim}. "
                    "Aborting to avoid data corruption. Recreate the collection or specify a new collection name."
                )
            return False

        # Create new collection
        try:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=expected_dim, distance=distance),
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection '{collection_name}': {e}") from e

        # Create payload indexes on filter fields if supported
        for field_name, field_type in [
            ("document_id", PayloadSchemaType.KEYWORD),
            ("physical_page_number", PayloadSchemaType.INTEGER),
        ]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
            except Exception:
                # Local embedded Qdrant emits a warning or ignores payload indexes
                pass

        return True

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert a batch of chunks and their embedding vectors into Qdrant.

        Args:
            collection_name: Name of target collection.
            chunks: List of DocumentChunk records.
            embeddings: List of embedding vectors corresponding to chunks.

        Returns:
            Number of points upserted.

        Raises:
            VectorStoreError: If chunks and embeddings lengths mismatch or upsert fails.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})"
            )

        if not chunks:
            return 0

        points: List[PointStruct] = []
        for chunk, vec in zip(chunks, embeddings):
            pt_id = self.chunk_id_to_point_id(chunk.chunk_id)
            payload: Dict[str, Any] = {
                "chunk_id": chunk.metadata.chunk_id,
                "document_id": chunk.metadata.document_id,
                "document_title": chunk.metadata.document_title,
                "physical_page_index": chunk.metadata.physical_page_index,
                "physical_page_number": chunk.metadata.physical_page_number,
                "printed_page_number": chunk.metadata.printed_page_number,
                "chunk_index_in_page": chunk.metadata.chunk_index_in_page,
                "chunk_index_in_doc": chunk.metadata.chunk_index_in_doc,
                "start_char_idx": chunk.metadata.start_char_idx,
                "end_char_idx": chunk.metadata.end_char_idx,
                "char_count": chunk.metadata.char_count,
                "word_count": chunk.metadata.word_count,
                "text": chunk.text,
            }
            points.append(PointStruct(id=pt_id, vector=vec, payload=payload))

        try:
            self._client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert batch of {len(points)} points into '{collection_name}': {e}"
            ) from e

        return len(points)

    def get_point_count(self, collection_name: str) -> int:
        """Get total number of points stored in a collection."""
        try:
            info = self._client.get_collection(collection_name)
            return info.points_count or 0
        except Exception as e:
            raise VectorStoreError(f"Failed to get point count for '{collection_name}': {e}") from e

    def get_point_by_chunk_id(self, collection_name: str, chunk_id: str) -> Optional[Record]:
        """Retrieve an individual point by its original chunk ID."""
        pt_id = self.chunk_id_to_point_id(chunk_id)
        try:
            records = self._client.retrieve(
                collection_name=collection_name,
                ids=[pt_id],
                with_payload=True,
                with_vectors=True,
            )
            return records[0] if records else None
        except Exception as e:
            raise VectorStoreError(f"Failed to retrieve point '{chunk_id}' from '{collection_name}': {e}") from e

