"""Indexing module for the Research Intelligence Engine.

Provides embedding generation, Qdrant vector storage, and indexing pipeline.
"""

from src.indexing.models import (
    IndexingConfig,
    IndexingSummary,
    IndexingError,
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
from src.indexing.vector_store import QdrantStoreManager
from src.indexing.pipeline import IndexingPipeline

__all__ = [
    "IndexingConfig",
    "IndexingSummary",
    "IndexingError",
    "EmbeddingError",
    "VectorStoreError",
    "IncompatibleCollectionError",
    "InvalidChunkRecordError",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "MockEmbedder",
    "QdrantStoreManager",
    "IndexingPipeline",
]

