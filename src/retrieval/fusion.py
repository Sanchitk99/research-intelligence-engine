"""Reciprocal Rank Fusion (RRF) for combining dense vector and BM25 keyword rankings."""

from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from src.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion(
    dense_chunks: List[RetrievedChunk],
    bm25_chunks: List[RetrievedChunk],
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int = 5,
    max_chunks_per_doc: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Fuse dense vector search and BM25 keyword search rankings using Reciprocal Rank Fusion.

    Computes RRF score for each unique chunk:
        RRF(c) = (dense_weight / (rrf_k + r_dense(c))) + (bm25_weight / (rrf_k + r_bm25(c)))

    Chunks appearing in only one candidate list receive the single reciprocal rank term.
    Preserves all citation provenance and metadata while recording individual subsystem ranks.
    Supports optional document diversification to prevent a single document from monopolizing
    the final context, with graceful score-ordered backfill.

    Args:
        dense_chunks: Ranked list of chunks from dense retrieval.
        bm25_chunks: Ranked list of chunks from BM25 retrieval.
        rrf_k: Smoothing constant (default 60, per Cormack et al. 2009).
        dense_weight: Multiplier weight for dense retrieval reciprocal ranks.
        bm25_weight: Multiplier weight for BM25 retrieval reciprocal ranks.
        top_k: Maximum number of fused chunks to return.
        max_chunks_per_doc: Optional limit on chunks from a single document. If None,
            standard RRF truncation is used.

    Returns:
        List of fused RetrievedChunk items ordered descending by RRF score.
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be an integer >= 1, got {rrf_k}")
    if top_k < 1:
        raise ValueError(f"top_k must be an integer >= 1, got {top_k}")
    if max_chunks_per_doc is not None and max_chunks_per_doc < 1:
        raise ValueError(f"max_chunks_per_doc must be an integer >= 1 or None, got {max_chunks_per_doc}")

    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_prototypes: Dict[str, RetrievedChunk] = {}
    dense_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    bm25_scores: Dict[str, float] = {}

    # 1. Process dense retrieval rankings
    for rank, chunk in enumerate(dense_chunks, 1):
        cid = chunk.chunk_id
        rrf_scores[cid] += dense_weight / (rrf_k + rank)
        dense_ranks[cid] = rank
        dense_scores[cid] = chunk.score
        chunk_prototypes[cid] = chunk

    # 2. Process BM25 retrieval rankings
    for rank, chunk in enumerate(bm25_chunks, 1):
        cid = chunk.chunk_id
        rrf_scores[cid] += bm25_weight / (rrf_k + rank)
        bm25_ranks[cid] = rank
        bm25_scores[cid] = chunk.score
        if cid not in chunk_prototypes:
            chunk_prototypes[cid] = chunk

    if not rrf_scores:
        return []

    # 3. Sort deterministically: descending by RRF score, secondary tie-break by chunk_id
    sorted_chunk_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: (-rrf_scores[cid], cid),
    )

    # 4. Select candidate chunk IDs (applying optional document diversification)
    if max_chunks_per_doc is None or max_chunks_per_doc < 1:
        selected_chunk_ids = sorted_chunk_ids[:top_k]
    else:
        selected_chunk_ids: List[str] = []
        selected_set = set()
        doc_counts: Dict[str, int] = defaultdict(int)

        # Pass 1: Select up to max_chunks_per_doc chunks per document in score order
        for cid in sorted_chunk_ids:
            proto = chunk_prototypes[cid]
            doc_id = proto.document_id
            if doc_counts[doc_id] < max_chunks_per_doc:
                selected_chunk_ids.append(cid)
                selected_set.add(cid)
                doc_counts[doc_id] += 1
                if len(selected_chunk_ids) == top_k:
                    break

        # Pass 2: Backfill remaining slots in score order from unselected candidates
        if len(selected_chunk_ids) < top_k:
            for cid in sorted_chunk_ids:
                if cid not in selected_set:
                    selected_chunk_ids.append(cid)
                    selected_set.add(cid)
                    if len(selected_chunk_ids) == top_k:
                        break

    # 5. Construct fused RetrievedChunk instances
    fused_chunks: List[RetrievedChunk] = []
    for cid in selected_chunk_ids:
        proto = chunk_prototypes[cid]
        final_score = round(rrf_scores[cid], 6)

        fused = RetrievedChunk(
            chunk_id=proto.chunk_id,
            document_id=proto.document_id,
            document_title=proto.document_title,
            physical_page_index=proto.physical_page_index,
            physical_page_number=proto.physical_page_number,
            printed_page_number=proto.printed_page_number,
            chunk_index_in_page=proto.chunk_index_in_page,
            chunk_index_in_doc=proto.chunk_index_in_doc,
            start_char_idx=proto.start_char_idx,
            end_char_idx=proto.end_char_idx,
            char_count=proto.char_count,
            word_count=proto.word_count,
            text=proto.text,
            score=final_score,
            retrieval_method="hybrid_rrf",
            dense_rank=dense_ranks.get(cid),
            bm25_rank=bm25_ranks.get(cid),
            dense_score=dense_scores.get(cid),
            bm25_score=bm25_scores.get(cid),
        )
        fused_chunks.append(fused)

    return fused_chunks

