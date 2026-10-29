"""BM25 keyword retrieval module for the Research Intelligence Engine.

Implements Okapi BM25 scoring with deterministic tokenization, inverted indexing,
document filtering, and full provenance preservation. Zero external dependencies.
"""

import json
import math
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from collections import Counter, defaultdict

from src.retrieval.models import (
    BM25Config,
    RetrievedChunk,
    RetrievalResult,
    RetrievalError,
    InvalidQueryError,
    FilterError,
)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens and decomposed hyphenated terms.

    Preserves compound terms (e.g. 'rag-sequence', 'bart-large', 'top-20')
    as well as their constituent parts to enable exact and partial matching.

    Args:
        text: Raw input text or query.

    Returns:
        List of lowercase string tokens.
    """
    raw_tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    tokens: List[str] = []
    for tok in raw_tokens:
        tokens.append(tok)
        if "-" in tok or "_" in tok:
            sub_tokens = re.split(r"[-_]", tok)
            for st in sub_tokens:
                if st:
                    tokens.append(st)
    return tokens


class BM25Index:
    """Inverted index and Okapi BM25 scoring engine for research paper chunks."""

    def __init__(
        self,
        chunks: List[RetrievedChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 index with pre-loaded chunks.

        Args:
            chunks: List of RetrievedChunk instances.
            k1: Term frequency saturation parameter (default 1.5).
            b: Document length normalization parameter (default 0.75).
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)

        # Precompute document tokenizations and lengths
        self.doc_tokens: List[List[str]] = [tokenize(c.text) for c in chunks]
        self.doc_lens: List[int] = [len(dt) for dt in self.doc_tokens]
        self.avgdl: float = sum(self.doc_lens) / max(1, self.N)

        # Build inverted index: term -> list of (doc_index, term_freq)
        self.df: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            for term, count in counts.items():
                self.df[term] += 1
                self.inverted_index[term].append((doc_idx, count))

        # Precompute Robertson-Spärck Jones IDF for all known terms: ln((N - n + 0.5)/(n + 0.5) + 1.0)
        self.idf: Dict[str, float] = {}
        for term, n_q in self.df.items():
            self.idf[term] = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)

    @classmethod
    def from_chunks_file(
        cls,
        json_path: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Load corpus chunks from JSON file and construct index.

        Args:
            json_path: Path to corpus_chunks.json.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.

        Returns:
            Configured BM25Index instance.
        """
        path = Path(json_path)
        if not path.is_file():
            raise RetrievalError(f"Corpus chunks file not found at '{path}'")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: List[RetrievedChunk] = []
        for item in data:
            meta = item.get("metadata", {})
            text = item.get("text", "")
            chunk = RetrievedChunk(
                chunk_id=str(item.get("chunk_id", meta.get("chunk_id", ""))),
                document_id=str(meta.get("document_id", "UNKNOWN")),
                document_title=str(meta.get("document_title", "")),
                physical_page_index=int(meta.get("physical_page_index", 0)),
                physical_page_number=int(meta.get("physical_page_number", 1)),
                printed_page_number=meta.get("printed_page_number"),
                chunk_index_in_page=int(meta.get("chunk_index_in_page", 0)),
                chunk_index_in_doc=int(meta.get("chunk_index_in_doc", 0)),
                start_char_idx=int(meta.get("start_char_idx", 0)),
                end_char_idx=int(meta.get("end_char_idx", len(text))),
                char_count=int(meta.get("char_count", len(text))),
                word_count=int(meta.get("word_count", len(text.split()))),
                text=text,
                score=0.0,
                retrieval_method="bm25",
            )
            chunks.append(chunk)

        return cls(chunks=chunks, k1=k1, b=b)

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_doc_indices: Optional[Set[int]] = None,
        min_score: Optional[float] = None,
    ) -> List[Tuple[RetrievedChunk, float]]:
        """Score and rank chunks for a natural language query using Okapi BM25.

        Args:
            query: Query string.
            top_k: Maximum number of ranked results to return.
            allowed_doc_indices: Optional set of document indices allowed by filters.
            min_score: Optional minimum BM25 score threshold.

        Returns:
            List of (RetrievedChunk, bm25_score) tuples ordered descending by score.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        # Accumulate scores using inverted index
        doc_scores: Dict[int, float] = defaultdict(float)

        for term in q_tokens:
            if term not in self.inverted_index:
                continue

            idf_t = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                if allowed_doc_indices is not None and doc_idx not in allowed_doc_indices:
                    continue

                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                numerator = tf * (self.k1 + 1.0)
                doc_scores[doc_idx] += idf_t * (numerator / denom)

        if not doc_scores:
            return []

        # Filter by min_score if specified
        candidates = [
            (idx, score)
            for idx, score in doc_scores.items()
            if min_score is None or score >= min_score
        ]

        # Sort descending by score; break ties by doc_idx (deterministic)
        candidates.sort(key=lambda x: (-x[1], x[0]))

        results: List[Tuple[RetrievedChunk, float]] = []
        for idx, score in candidates[:top_k]:
            results.append((self.chunks[idx], score))

        return results


class BM25Retriever:
    """Retriever component implementing BM25 keyword search over indexed chunks."""

    def __init__(
        self,
        config: Optional[BM25Config] = None,
        index: Optional[BM25Index] = None,
    ) -> None:
        """Initialize BM25 retriever.

        Args:
            config: BM25 configuration settings.
            index: Optional pre-built BM25Index instance. Lazily built if None.
        """
        self.config = config or BM25Config()
        self._index = index

    @property
    def index(self) -> BM25Index:
        """Get or lazily load BM25 index."""
        if self._index is None:
            self._index = BM25Index.from_chunks_file(
                json_path=self.config.corpus_chunks_path,
                k1=self.config.k1,
                b=self.config.b,
            )
        return self._index

    @staticmethod
    def validate_query(query: str) -> str:
        """Validate input query string."""
        if not isinstance(query, str):
            raise InvalidQueryError(f"Query must be a string, got {type(query).__name__}")
        cleaned = query.strip()
        if not cleaned:
            raise InvalidQueryError("Query cannot be empty or whitespace-only")
        return cleaned

    @staticmethod
    def build_document_filter(
        document_ids: Optional[Union[str, List[str]]],
    ) -> Tuple[Optional[Set[str]], Optional[List[str]]]:
        """Construct normalized document ID filter set."""
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

        return set(cleaned_ids), cleaned_ids

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[Union[str, List[str]]] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve most relevant chunks using BM25 keyword scoring.

        Args:
            query: Natural language question or search query.
            top_k: Maximum number of chunks to return (overrides config if provided).
            document_ids: Optional document ID or list of document IDs to restrict search.
            min_score: Optional minimum BM25 score threshold.

        Returns:
            RetrievalResult containing ranked RetrievedChunk items and metadata.

        Raises:
            InvalidQueryError: If query is invalid or empty.
            FilterError: If document filter list is invalid.
            RetrievalError: If retrieval fails.
        """
        start_time = time.perf_counter()
        cleaned_query = self.validate_query(query)

        target_top_k = top_k if top_k is not None else self.config.top_k
        if not isinstance(target_top_k, int) or target_top_k < 1:
            raise RetrievalError(f"top_k must be an integer >= 1, got {target_top_k}")

        doc_id_set, norm_doc_ids = self.build_document_filter(document_ids)

        # Map document_ids filter to allowed chunk indices
        allowed_indices: Optional[Set[int]] = None
        if doc_id_set is not None:
            allowed_indices = {
                idx
                for idx, chunk in enumerate(self.index.chunks)
                if chunk.document_id in doc_id_set
            }
            if not allowed_indices:
                # Document ID does not match any corpus chunk
                return RetrievalResult(
                    query=cleaned_query,
                    chunks=[],
                    total_retrieved=0,
                    collection_name="bm25_corpus",
                    document_filter=norm_doc_ids,
                    execution_time_ms=0.0,
                )

        ranked_matches = self.index.search(
            query=cleaned_query,
            top_k=target_top_k,
            allowed_doc_indices=allowed_indices,
            min_score=min_score,
        )

        retrieved_chunks: List[RetrievedChunk] = []
        for rank, (src_chunk, score) in enumerate(ranked_matches, 1):
            chunk = RetrievedChunk(
                chunk_id=src_chunk.chunk_id,
                document_id=src_chunk.document_id,
                document_title=src_chunk.document_title,
                physical_page_index=src_chunk.physical_page_index,
                physical_page_number=src_chunk.physical_page_number,
                printed_page_number=src_chunk.printed_page_number,
                chunk_index_in_page=src_chunk.chunk_index_in_page,
                chunk_index_in_doc=src_chunk.chunk_index_in_doc,
                start_char_idx=src_chunk.start_char_idx,
                end_char_idx=src_chunk.end_char_idx,
                char_count=src_chunk.char_count,
                word_count=src_chunk.word_count,
                text=src_chunk.text,
                score=round(score, 5),
                retrieval_method="bm25",
                bm25_rank=rank,
                bm25_score=round(score, 5),
            )
            retrieved_chunks.append(chunk)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=cleaned_query,
            chunks=retrieved_chunks,
            total_retrieved=len(retrieved_chunks),
            collection_name="bm25_corpus",
            document_filter=norm_doc_ids,
            execution_time_ms=round(elapsed_ms, 2),
            bm25_time_ms=round(elapsed_ms, 2),
        )

    def close(self) -> None:
        """No-op resource cleanup for interface consistency."""
        pass

    def __enter__(self) -> "BM25Retriever":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

