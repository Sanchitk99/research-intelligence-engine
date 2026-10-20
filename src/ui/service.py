"""Engine service layer coordinating ingestion, indexing, retrieval, and grounded Q&A.

Provides a unified, decoupled interface for both UI frontends (Streamlit)
and headless demonstration scripts.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from src.ingestion.loader import PDFLoader
from src.ingestion.models import DocumentMetadata, DocumentIngestionResult
from src.chunking.text_chunker import TextChunker
from src.chunking.models import ChunkingConfig, DocumentChunk
from src.indexing.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from src.indexing.vector_store import QdrantStoreManager
from src.indexing.pipeline import IndexingPipeline
from src.indexing.models import IndexingConfig
from src.retrieval.retriever import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievedChunk
from src.generation.generator import AnswerGenerator
from src.generation.models import GeneratedAnswer, AnswerStatus, GenerationConfig
from src.generation.providers.factory import get_llm_provider
from src.generation.providers.base import (
    BaseLLMProvider,
    ProviderError,
    ProviderAuthError,
    ProviderTimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperSummary:
    """Summary of an indexed research paper."""
    document_id: str
    title: str
    total_pages: int
    total_chunks: int
    filename: Optional[str] = None


@dataclass
class IngestionReport:
    """Report produced after ingesting and indexing a paper."""
    success: bool
    document_id: str
    title: str
    filename: str
    pages_extracted: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    total_collection_points: int = 0
    error_message: Optional[str] = None
    sample_chunks: List[Dict[str, Any]] = field(default_factory=list)


class ResearchEngineService:
    """Unified service orchestrating the Research Intelligence Engine workflow."""

    def __init__(
        self,
        qdrant_path: Union[str, Path] = "data/qdrant",
        collection_name: str = "research_corpus",
        chunks_file: Union[str, Path] = "data/chunked/corpus_chunks.json",
        embedder: Optional[BaseEmbedder] = None,
        store_manager: Optional[QdrantStoreManager] = None,
    ) -> None:
        """Initialize the engine service with paths and shared components."""
        self.qdrant_path = Path(qdrant_path)
        self.collection_name = collection_name
        self.chunks_file = Path(chunks_file)

        # Lazy components
        self._embedder = embedder
        self._store_manager = store_manager
        self._owns_store = store_manager is None
        self._loader: Optional[PDFLoader] = None
        self._chunker: Optional[TextChunker] = None

    @property
    def embedder(self) -> BaseEmbedder:
        """Lazily initialize shared embedding model."""
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                normalize_embeddings=True,
            )
        return self._embedder

    @property
    def store_manager(self) -> QdrantStoreManager:
        """Lazily initialize shared Qdrant store manager."""
        if self._store_manager is None:
            self._store_manager = QdrantStoreManager(qdrant_path=self.qdrant_path)
        return self._store_manager

    @property
    def loader(self) -> PDFLoader:
        """Lazily initialize PDF loader."""
        if self._loader is None:
            self._loader = PDFLoader(raise_on_error=False)
        return self._loader

    @property
    def chunker(self) -> TextChunker:
        """Lazily initialize text chunker."""
        if self._chunker is None:
            self._chunker = TextChunker()
        return self._chunker

    def close(self) -> None:
        """Close shared resources."""
        if self._owns_store and self._store_manager is not None:
            self._store_manager.close()
            self._store_manager = None

    def __enter__(self) -> "ResearchEngineService":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_system_status(self) -> Dict[str, Any]:
        """Query system health, indexed point count, and configured providers."""
        gemini_key_set = bool(os.getenv("GEMINI_API_KEY", "").strip())
        openai_key_set = bool(os.getenv("OPENAI_API_KEY", "").strip())
        groq_key_set = bool(os.getenv("GROQ_API_KEY", "").strip())

        try:
            point_count = self.store_manager.get_point_count(self.collection_name)
        except Exception:
            point_count = 0

        papers = self.get_indexed_papers()

        return {
            "collection_name": self.collection_name,
            "total_points": point_count,
            "indexed_papers_count": len(papers),
            "papers": papers,
            "providers": {
                "gemini": {"configured": gemini_key_set, "default_model": "gemini-flash-lite-latest"},
                "openai": {"configured": openai_key_set, "default_model": "gpt-4o-mini"},
                "groq": {"configured": groq_key_set, "default_model": "llama-3.3-70b-versatile"},
                "mock": {"configured": True, "default_model": "mock-evaluator"},
                "ollama": {"configured": True, "default_model": "mistral"},
            },
        }

    def get_indexed_papers(self) -> List[PaperSummary]:
        """Scan corpus_chunks.json and return summaries of all indexed papers."""
        if not self.chunks_file.is_file():
            return []

        try:
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read chunks file '{self.chunks_file}': {e}")
            return []

        if not isinstance(data, list):
            return []

        doc_stats: Dict[str, Dict[str, Any]] = {}
        for item in data:
            meta = item.get("metadata", {})
            doc_id = meta.get("document_id") or "UNKNOWN"
            doc_title = meta.get("document_title") or doc_id
            page_num = meta.get("physical_page_number") or 1

            if doc_id not in doc_stats:
                doc_stats[doc_id] = {
                    "document_id": doc_id,
                    "title": doc_title,
                    "max_page": page_num,
                    "chunks_count": 0,
                }
            doc_stats[doc_id]["chunks_count"] += 1
            if page_num > doc_stats[doc_id]["max_page"]:
                doc_stats[doc_id]["max_page"] = page_num

        summaries = [
            PaperSummary(
                document_id=info["document_id"],
                title=info["title"],
                total_pages=info["max_page"],
                total_chunks=info["chunks_count"],
            )
            for info in doc_stats.values()
        ]
        return sorted(summaries, key=lambda p: p.document_id)

    def ingest_paper(
        self,
        pdf_path: Union[str, Path],
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        dry_run: bool = False,
    ) -> IngestionReport:
        """Execute the end-to-end ingestion pipeline on a single PDF document.

        1. PDF text and metadata extraction (page-by-page).
        2. Boundary-aware text chunking.
        3. Embedding generation and Qdrant point upsert.
        4. Synchronizing local corpus_chunks.json for BM25 retrieval.

        Args:
            pdf_path: Path to the target PDF file.
            doc_id: Optional document identifier override (e.g. 'P6').
            doc_title: Optional document title override.
            dry_run: If True, validate extraction and chunking without persisting.

        Returns:
            IngestionReport detailing the extraction and indexing counts.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            return IngestionReport(
                success=False,
                document_id=doc_id or path.stem,
                title=doc_title or path.stem,
                filename=path.name,
                error_message=f"PDF file does not exist: {path}",
            )

        resolved_id = (doc_id or path.stem).strip()
        resolved_title = (doc_title or path.stem.replace("_", " ")).strip()

        # Step 1: PDF Extraction
        metadata = DocumentMetadata(
            id=resolved_id,
            title=resolved_title,
            filename=path.name,
        )
        ingest_result: DocumentIngestionResult = self.loader.load_document(path, metadata=metadata)
        if not ingest_result.success or not ingest_result.pages:
            return IngestionReport(
                success=False,
                document_id=resolved_id,
                title=resolved_title,
                filename=path.name,
                error_message=ingest_result.error_message or "No pages could be extracted from PDF.",
            )

        # Step 2: Text Chunking
        chunks: List[DocumentChunk] = self.chunker.chunk_document_pages(ingest_result.pages)
        if not chunks:
            return IngestionReport(
                success=False,
                document_id=resolved_id,
                title=resolved_title,
                filename=path.name,
                pages_extracted=len(ingest_result.pages),
                error_message="Document text was chunked into 0 chunks (empty text).",
            )

        # Step 3: Embed and Index into Qdrant
        chunks_indexed = 0
        total_points = 0
        if not dry_run:
            indexing_pipe = IndexingPipeline(
                config=IndexingConfig(
                    collection_name=self.collection_name,
                    qdrant_path=str(self.qdrant_path),
                ),
                embedder=self.embedder,
                store_manager=self.store_manager,
            )
            idx_summary = indexing_pipe.index_chunks(chunks, dry_run=False)
            chunks_indexed = idx_summary.chunks_indexed
            total_points = idx_summary.total_points_in_collection

            # Step 4: Synchronize corpus_chunks.json for BM25
            self._sync_chunks_to_file(chunks, resolved_id)
        else:
            chunks_indexed = len(chunks)
            total_points = self.store_manager.get_point_count(self.collection_name)

        sample_previews = [
            {
                "chunk_id": c.chunk_id,
                "page": c.metadata.physical_page_number,
                "words": c.metadata.word_count,
                "preview": c.text[:120].strip() + ("..." if len(c.text) > 120 else ""),
            }
            for c in chunks[:5]
        ]

        return IngestionReport(
            success=True,
            document_id=resolved_id,
            title=resolved_title,
            filename=path.name,
            pages_extracted=len(ingest_result.pages),
            chunks_created=len(chunks),
            chunks_indexed=chunks_indexed,
            total_collection_points=total_points,
            sample_chunks=sample_previews,
        )

    def _sync_chunks_to_file(self, new_chunks: List[DocumentChunk], doc_id: str) -> None:
        """Safely synchronize new chunks into corpus_chunks.json, creating a backup first."""
        self.chunks_file.parent.mkdir(parents=True, exist_ok=True)

        existing_data: List[Dict[str, Any]] = []
        if self.chunks_file.is_file():
            # Create a backup on first update if backup doesn't exist
            bak_path = self.chunks_file.with_suffix(".json.bak")
            if not bak_path.exists():
                try:
                    shutil.copy2(self.chunks_file, bak_path)
                except Exception:
                    pass

            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        # Filter out any chunks with matching doc_id to avoid duplication
                        existing_data = [
                            c for c in content
                            if c.get("metadata", {}).get("document_id") != doc_id
                        ]
            except Exception as e:
                logger.warning(f"Could not read existing chunks file: {e}")

        # Convert new chunks to dict records
        serialized_new = [c.model_dump() for c in new_chunks]
        combined = existing_data + serialized_new

        tmp_path = self.chunks_file.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.chunks_file)

    def ask_question(
        self,
        question: str,
        provider_name: str = "auto",
        model_name: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> GeneratedAnswer:
        """Ask a research question and generate an evidence-grounded answer with citations.

        Args:
            question: Natural language question about the corpus.
            provider_name: 'auto', 'gemini', 'mock', 'openai', 'groq', or 'ollama'.
            model_name: Optional model override string.
            retrieval_mode: 'hybrid' or 'dense'.
            top_k: Number of chunks to retrieve (default: 5).
            min_score: Optional similarity cutoff.

        Returns:
            GeneratedAnswer with verified citations, answer status, and latency.
        """
        start_time = time.perf_counter()

        # Sanitize query
        q_cleaned = (question or "").strip()
        if not q_cleaned:
            return GeneratedAnswer(
                question="",
                answer="Question cannot be empty or whitespace-only.",
                status=AnswerStatus.ERROR,
                retrieved_chunks=[],
                model_name=model_name or "unknown",
                provider_name=provider_name,
                retrieval_mode=retrieval_mode,
                latency_ms=0.0,
            )

        try:
            # 1. Resolve LLM provider safely
            provider: BaseLLMProvider = get_llm_provider(
                provider_type=provider_name,
                model_name=model_name,
            )

            # 2. Build retrieval component sharing pre-loaded embedder and store_manager
            dense_retriever = DenseRetriever(
                embedder=self.embedder,
                store_manager=self.store_manager,
            )

            if retrieval_mode.lower() == "hybrid":
                retriever = HybridRetriever(
                    dense_retriever=dense_retriever,
                )
            else:
                retriever = dense_retriever

            # 3. Configure and run AnswerGenerator
            config = GenerationConfig(
                provider_type=provider.name,
                model_name=provider.default_model,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                min_score=min_score,
            )

            generator = AnswerGenerator(
                retriever=retriever,
                provider=provider,
                config=config,
                retrieval_mode=retrieval_mode,
            )

            result = generator.generate(
                question=q_cleaned,
                top_k=top_k,
                min_score=min_score,
            )
            return result

        except ProviderAuthError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer=(
                    f"Authentication failed for provider '{provider_name}'. "
                    "Please ensure the required API key is set in your environment or .env file."
                ),
                status=AnswerStatus.ERROR,
                retrieved_chunks=[],
                model_name=model_name or "unknown",
                provider_name=provider_name,
                retrieval_mode=retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
                confidence_notes="Provider authentication error. Credentials were not exposed.",
            )
        except ProviderTimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer=(
                    f"The request to provider '{provider_name}' timed out. "
                    "Please check your internet connection or try again."
                ),
                status=AnswerStatus.ERROR,
                retrieved_chunks=[],
                model_name=model_name or "unknown",
                provider_name=provider_name,
                retrieval_mode=retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
                confidence_notes="Provider request timed out.",
            )
        except RateLimitError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer=(
                    f"The rate limit for provider '{provider_name}' was exceeded. "
                    "Please wait a few moments before submitting another question."
                ),
                status=AnswerStatus.ERROR,
                retrieved_chunks=[],
                model_name=model_name or "unknown",
                provider_name=provider_name,
                retrieval_mode=retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
                confidence_notes="Rate limit reached.",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Sanitize error message to prevent any secret or raw traceback leakage
            clean_err = str(e).split("\n")[0][:120]
            return GeneratedAnswer(
                question=q_cleaned,
                answer=(
                    f"An error occurred while processing your request: {clean_err}. "
                    "Please verify your engine configuration."
                ),
                status=AnswerStatus.ERROR,
                retrieved_chunks=[],
                model_name=model_name or "unknown",
                provider_name=provider_name,
                retrieval_mode=retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
                confidence_notes="Execution error safely caught.",
            )

