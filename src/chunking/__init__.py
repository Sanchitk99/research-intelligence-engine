"""Text Chunking Module for Research Intelligence Engine."""

from src.chunking.models import (
    ChunkingConfig,
    ChunkMetadata,
    DocumentChunk,
    ChunkingSummary,
)
from src.chunking.text_chunker import TextChunker
from src.chunking.pipeline import chunk_corpus

__all__ = [
    "ChunkingConfig",
    "ChunkMetadata",
    "DocumentChunk",
    "ChunkingSummary",
    "TextChunker",
    "chunk_corpus",
]

