"""Hybrid retriever combining dense vector retrieval and BM25 keyword retrieval using RRF."""

import time
from typing import Optional, List, Union

from src.retrieval.models import (
    HybridRetrievalConfig,
    RetrievalConfig,
    BM25Config,
    RetrievedChunk,
    RetrievalResult,
    RetrievalError,
    InvalidQueryError,
    FilterError,
)
from src.retrieval.retriever import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.router import classify_query_intent, QueryIntent


class HybridRetriever:
    """Orchestrates hybrid retrieval combining dense vector search and BM25 with Reciprocal Rank Fusion."""

    def __init__(
        self,
        config: Optional[HybridRetrievalConfig] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            config: Hybrid retrieval configuration settings.
            dense_retriever: Optional DenseRetriever instance. Lazily initialized if None.
            bm25_retriever: Optional BM25Retriever instance. Lazily initialized if None.
        """
        self.config = config or HybridRetrievalConfig()

        dense_cfg = self.config.dense_config or RetrievalConfig(
            collection_name="research_corpus",
            top_k=self.config.candidate_k,
        )
        bm25_cfg = self.config.bm25_config or BM25Config(
            top_k=self.config.candidate_k,
        )

        self.dense_retriever: DenseRetriever = dense_retriever or DenseRetriever(config=dense_cfg)
        self.bm25_retriever: BM25Retriever = bm25_retriever or BM25Retriever(config=bm25_cfg)
        self._owns_dense = dense_retriever is None

    def close(self) -> None:
        """Safely close underlying dense vector store connections if owned."""
        if self._owns_dense:
            self.dense_retriever.close()
        self.bm25_retriever.close()

    def __enter__(self) -> "HybridRetriever":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[Union[str, List[str]]] = None,
        min_score: Optional[float] = None,
        max_chunks_per_doc: Optional[int] = None,
        enable_intent_routing: Optional[bool] = None,
    ) -> RetrievalResult:
        """Retrieve and fuse candidate chunks from dense and BM25 subsystems.

        Args:
            query: Natural language research question or search query.
            top_k: Final number of fused chunks to return (default from config).
            document_ids: Optional document ID or list of document IDs to restrict search.
            min_score: Optional minimum score threshold for RRF scores.
            max_chunks_per_doc: Optional per-document limit to enforce diversification.
                If None and intent routing is disabled, uses self.config.max_chunks_per_doc (default None).
            enable_intent_routing: Optional toggle for query-intent routing. If True,
                detects comparative vs standard queries and chooses diversification accordingly.
                If None, defaults to self.config.enable_intent_routing (default False).

        Returns:
            RetrievalResult containing deduplicated, RRF-ranked RetrievedChunk items.

        Raises:
            InvalidQueryError: If query is empty or whitespace-only.
            FilterError: If document_ids filter specification is invalid.
            RetrievalError: If retrieval or fusion fails.
        """
        start_time = time.perf_counter()

        # Validate query string
        cleaned_query = DenseRetriever.validate_query(query)

        target_top_k = top_k if top_k is not None else self.config.top_k
        if not isinstance(target_top_k, int) or target_top_k < 1:
            raise RetrievalError(f"top_k must be an integer >= 1, got {target_top_k}")

        # Determine if query-intent routing should be applied
        routing_active = (
            enable_intent_routing
            if enable_intent_routing is not None
            else self.config.enable_intent_routing
        )

        detected_intent: Optional[str] = None
        if routing_active:
            intent_res = classify_query_intent(cleaned_query)
            detected_intent = intent_res.intent.value

            if intent_res.intent == QueryIntent.COMPARATIVE:
                # Use comparative diversification cap (default 2 for top-k <= 5)
                if target_top_k <= 5:
                    effective_max_per_doc = self.config.comparative_max_chunks_per_doc
                else:
                    effective_max_per_doc = max(self.config.comparative_max_chunks_per_doc + 1, 3)
            else:
                # STANDARD intent: retain default unconstrained baseline
                effective_max_per_doc = (
                    max_chunks_per_doc
                    if max_chunks_per_doc is not None
                    else self.config.max_chunks_per_doc
                )
        else:
            effective_max_per_doc = (
                max_chunks_per_doc
                if max_chunks_per_doc is not None
                else self.config.max_chunks_per_doc
            )

        if effective_max_per_doc is not None and (not isinstance(effective_max_per_doc, int) or effective_max_per_doc < 1):
            raise RetrievalError(f"max_chunks_per_doc must be None or an integer >= 1, got {effective_max_per_doc}")

        candidate_k = max(target_top_k, self.config.candidate_k)

        # 1. Execute Dense Vector Retrieval
        dense_res = self.dense_retriever.retrieve(
            query=cleaned_query,
            top_k=candidate_k,
            document_ids=document_ids,
        )

        # 2. Execute BM25 Keyword Retrieval
        bm25_res = self.bm25_retriever.retrieve(
            query=cleaned_query,
            top_k=candidate_k,
            document_ids=document_ids,
        )

        # 3. Fuse via Reciprocal Rank Fusion (RRF) with optional diversification
        fused_chunks = reciprocal_rank_fusion(
            dense_chunks=dense_res.chunks,
            bm25_chunks=bm25_res.chunks,
            rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            bm25_weight=self.config.bm25_weight,
            top_k=target_top_k,
            max_chunks_per_doc=effective_max_per_doc,
        )

        # Apply optional minimum score filter
        if min_score is not None:
            fused_chunks = [c for c in fused_chunks if c.score >= min_score]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=cleaned_query,
            chunks=fused_chunks,
            total_retrieved=len(fused_chunks),
            collection_name=f"{dense_res.collection_name}+bm25",
            document_filter=dense_res.document_filter,
            execution_time_ms=round(elapsed_ms, 2),
            dense_time_ms=dense_res.execution_time_ms,
            bm25_time_ms=bm25_res.execution_time_ms,
            query_intent=detected_intent,
        )

