"""Dense retriever searching persistent Qdrant vector store with sentence embeddings.

Embeds queries with the configured embedding model, executes cosine similarity search,
applies optional document filtering, and returns citation-grounded results.
"""

import time
from typing import List, Optional, Union, Dict, Any, Tuple

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    ScoredPoint,
)

from src.indexing.embeddings import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
)
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


class DenseRetriever:
    """Performs dense vector retrieval against a local Qdrant collection."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
        store_manager: Optional[QdrantStoreManager] = None,
    ) -> None:
        """Initialize dense retriever.

        Args:
            config: Retrieval configuration settings.
            embedder: Optional BaseEmbedder instance (defaults to SentenceTransformerEmbedder).
            store_manager: Optional QdrantStoreManager instance (defaults to local QdrantStoreManager).
        """
        self.config = config or RetrievalConfig()
        self.embedder: BaseEmbedder = embedder or SentenceTransformerEmbedder(
            model_name=self.config.model_name,
            normalize_embeddings=True,
        )
        self.store_manager: QdrantStoreManager = store_manager or QdrantStoreManager(
            qdrant_path=self.config.qdrant_path
        )
        self._owns_store = store_manager is None

    def close(self) -> None:
        """Safely close vector store resources if owned."""
        if self._owns_store:
            self.store_manager.close()

    def __enter__(self) -> "DenseRetriever":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def validate_query(query: str) -> str:
        """Validate input query string.

        Args:
            query: Raw user query string.

        Returns:
            Cleaned query string.

        Raises:
            InvalidQueryError: If query is not a string, or is empty / whitespace-only.
        """
        if not isinstance(query, str):
            raise InvalidQueryError(f"Query must be a string, got {type(query).__name__}")
        cleaned = query.strip()
        if not cleaned:
            raise InvalidQueryError("Query cannot be empty or whitespace-only")
        return cleaned

    @staticmethod
    def build_document_filter(
        document_ids: Optional[Union[str, List[str]]],
    ) -> Tuple[Optional[Filter], Optional[List[str]]]:
        """Construct a Qdrant filter targeting specific document IDs.

        Args:
            document_ids: Single document ID string or list of document IDs.

        Returns:
            Tuple of (Qdrant Filter object or None, normalized list of doc IDs or None).

        Raises:
            FilterError: If document_ids contains invalid or empty values.
        """
        if document_ids is None:
            return None, None

        if isinstance(document_ids, str):
            cleaned_ids = [document_ids.strip()]
        elif isinstance(document_ids, (list, tuple)):
            cleaned_ids = [str(d).strip() for d in document_ids if str(d).strip()]
        else:
            raise FilterError(f"document_ids must be str or list of str, got {type(document_ids).__name__}")

        if not cleaned_ids:
            raise FilterError("Document filter list contains no valid non-empty document IDs")

        if len(cleaned_ids) == 1:
            q_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=cleaned_ids[0]))])
        else:
            q_filter = Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=cleaned_ids))])

        return q_filter, cleaned_ids

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[Union[str, List[str]]] = None,
        min_score: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve most relevant chunks for a research question.

        Args:
            query: Natural language question or search query.
            top_k: Maximum number of chunks to return (overrides config if provided).
            document_ids: Optional document ID or list of document IDs to restrict search.
            min_score: Optional minimum similarity threshold.
            collection_name: Optional collection name override.

        Returns:
            RetrievalResult containing ranked RetrievedChunk items and metadata.

        Raises:
            InvalidQueryError: If query is invalid or empty.
            CollectionNotFoundError: If target collection does not exist in Qdrant.
            RetrievalError: If vector search or embedding generation fails.
        """
        start_time = time.perf_counter()

        cleaned_query = self.validate_query(query)
        target_top_k = top_k if top_k is not None else self.config.top_k
        if not isinstance(target_top_k, int) or target_top_k < 1:
            raise RetrievalError(f"top_k must be an integer >= 1, got {target_top_k}")

        target_collection = collection_name or self.config.collection_name
        score_threshold = min_score if min_score is not None else self.config.min_score

        # Ensure target collection exists
        client = self.store_manager.client
        try:
            if not client.collection_exists(target_collection):
                raise CollectionNotFoundError(
                    f"Collection '{target_collection}' does not exist in Qdrant store at '{self.config.qdrant_path}'"
                )
        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise RetrievalError(f"Error checking collection '{target_collection}': {e}") from e

        # Build optional document filter
        q_filter, norm_doc_ids = self.build_document_filter(document_ids)

        # Generate dense query embedding vector
        try:
            query_vector = self.embedder.embed_text(cleaned_query)
        except Exception as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e

        # Search collection via Qdrant
        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=target_collection,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=target_top_k,
                    with_payload=True,
                )
                raw_points = response.points
            elif hasattr(client, "search"):
                raw_points = client.search(
                    collection_name=target_collection,
                    query_vector=query_vector,
                    query_filter=q_filter,
                    limit=target_top_k,
                    with_payload=True,
                )
            else:
                raise RetrievalError("QdrantClient does not support query_points or search methods")
        except Exception as e:
            raise RetrievalError(f"Vector search failed on collection '{target_collection}': {e}") from e

        retrieved_chunks: List[RetrievedChunk] = []
        for pt in raw_points:
            score = float(pt.score)
            if score_threshold is not None and score < score_threshold:
                continue

            payload = pt.payload or {}
            chunk = RetrievedChunk(
                chunk_id=str(payload.get("chunk_id", str(pt.id))),
                document_id=str(payload.get("document_id", "UNKNOWN")),
                document_title=str(payload.get("document_title", "")),
                physical_page_index=int(payload.get("physical_page_index", 0)),
                physical_page_number=int(payload.get("physical_page_number", 1)),
                printed_page_number=payload.get("printed_page_number"),
                chunk_index_in_page=int(payload.get("chunk_index_in_page", 0)),
                chunk_index_in_doc=int(payload.get("chunk_index_in_doc", 0)),
                start_char_idx=int(payload.get("start_char_idx", 0)),
                end_char_idx=int(payload.get("end_char_idx", len(str(payload.get("text", ""))))),
                char_count=int(payload.get("char_count", len(str(payload.get("text", ""))))),
                word_count=int(payload.get("word_count", len(str(payload.get("text", "")).split()))),
                text=str(payload.get("text", "")),
                score=round(score, 5),
            )
            retrieved_chunks.append(chunk)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=cleaned_query,
            chunks=retrieved_chunks,
            total_retrieved=len(retrieved_chunks),
            collection_name=target_collection,
            document_filter=norm_doc_ids,
            execution_time_ms=round(elapsed_ms, 2),
        )
