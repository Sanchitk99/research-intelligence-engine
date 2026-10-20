"""Data models and configuration for the indexing module.

Maintains configuration schemas and summary metrics for embedding and vector storage.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class IndexingError(Exception):
    """Base exception for all indexing pipeline errors."""
    pass


class EmbeddingError(IndexingError):
    """Raised when text embedding generation fails or produces invalid vectors."""
    pass


class VectorStoreError(IndexingError):
    """Raised when vector database operations fail."""
    pass


class IncompatibleCollectionError(VectorStoreError):
    """Raised when an existing collection has vector dimensions or metrics incompatible with the model."""
    pass


class InvalidChunkRecordError(IndexingError):
    """Raised when an input chunk record is malformed or invalid."""
    pass


class IndexingConfig(BaseModel):
    """Configuration settings for embedding generation and local Qdrant indexing."""
    model_config = ConfigDict(extra="ignore")

    collection_name: str = Field(
        default="research_corpus",
        description="Name of the Qdrant collection for research chunks"
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model identifier for SentenceTransformer embeddings"
    )
    qdrant_path: Path = Field(
        default=Path("data/qdrant"),
        description="Persistent filesystem directory for local Qdrant storage"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Number of chunks to embed and upsert in each batch"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to apply L2 normalization to output embeddings"
    )


class IndexingSummary(BaseModel):
    """Summary report detailing the results of an indexing execution."""
    model_config = ConfigDict(extra="ignore")

    chunks_read: int = Field(default=0, description="Total chunk records read from input")
    chunks_embedded: int = Field(default=0, description="Total chunks successfully embedded")
    chunks_indexed: int = Field(default=0, description="Total points upserted to Qdrant")
    chunks_skipped: int = Field(default=0, description="Chunks skipped (e.g. whitespace-only)")
    chunks_failed: int = Field(default=0, description="Chunks that failed validation or indexing")
    model_name: str = Field(default="", description="Embedding model name used")
    vector_dim: int = Field(default=0, description="Embedding vector dimensionality")
    collection_name: str = Field(default="", description="Target Qdrant collection name")
    total_points_in_collection: int = Field(
        default=0,
        description="Total point count in Qdrant collection after indexing"
    )
    dry_run: bool = Field(default=False, description="Whether execution was dry-run only")
    execution_time_seconds: float = Field(default=0.0, description="Total execution time in seconds")

