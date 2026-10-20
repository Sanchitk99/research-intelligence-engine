"""Dense retrieval module for the Research Intelligence Engine.

Provides DenseRetriever for embedding queries, searching Qdrant collections,
and returning ranked chunks with complete citation provenance.
"""

from src.retrieval.models import (
    RetrievalConfig,
    BM25Config,
    HybridRetrievalConfig,
    RetrievedChunk,
    RetrievalResult,
    RetrievalError,
    InvalidQueryError,
    CollectionNotFoundError,
    FilterError,
)
from src.retrieval.retriever import DenseRetriever
from src.retrieval.bm25 import BM25Index, BM25Retriever, tokenize
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever

__all__ = [
    "RetrievalConfig",
    "BM25Config",
    "HybridRetrievalConfig",
    "RetrievedChunk",
    "RetrievalResult",
    "RetrievalError",
    "InvalidQueryError",
    "CollectionNotFoundError",
    "FilterError",
    "DenseRetriever",
    "BM25Index",
    "BM25Retriever",
    "HybridRetriever",
    "tokenize",
    "reciprocal_rank_fusion",
]

