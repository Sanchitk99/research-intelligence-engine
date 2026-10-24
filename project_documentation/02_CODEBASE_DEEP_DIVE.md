# 02 — Codebase Deep Dive: Module-by-Module Technical Analysis

This document provides an exhaustive, file-by-file technical breakdown of every core source file in `src/`. For each file, we document its architectural responsibility, imports, key classes, functions, data flow, error handling, design decisions, and edge cases.

---

## 1. Document Ingestion Module (`src/ingestion/`)

### 1.1 `src/ingestion/models.py`
* **Responsibility:** Defines data transfer models for raw document ingestion, extracted page metadata, and batch ingestion summaries using Pydantic v2.
* **Key Classes:**
  * `DocumentMetadata(BaseModel)`: Captures document-level provenance. Fields: `id: str` (e.g., 'P1'), `title: str`, `filename: str`, `year: Optional[int]`, `authors: Optional[List[str]]`, `total_pages: Optional[int]`.
  * `PageMetadata(BaseModel)`: Captures page-level provenance. Fields: `document_id: str`, `physical_page_index: int` (0-based array index), `physical_page_number: int` (1-based physical sheet number), `printed_page_number: Optional[int]` (number printed in header/footer), `char_count: int`, `word_count: int`.
  * `DocumentPage(BaseModel)`: Encapsulates page text and metadata. Fields: `text: str`, `metadata: PageMetadata`.
  * `DocumentIngestionResult(BaseModel)`: Container for document extraction results. Fields: `document_metadata: DocumentMetadata`, `pages: List[DocumentPage]`, `success: bool`, `error_message: Optional[str]`.
  * `BatchIngestionSummary(BaseModel)`: Summary for batch operations. Fields: `total_documents: int`, `successful_documents: int`, `failed_documents: int`, `total_pages: int`, `results: List[DocumentIngestionResult]`.
* **Design Decisions:**
  * Strict separation between `physical_page_index` (0-based for code loops), `physical_page_number` (1-based for human citation and PDF reader jump), and `printed_page_number` (may differ due to roman numeral front matter).
* **Edge Cases:** Empty PDF extraction returns `success=False` with descriptive `error_message`.

### 1.2 `src/ingestion/loader.py`
* **Responsibility:** Implements `PDFLoader`, wrapping PyMuPDF (`fitz`) to extract text and structural metadata from PDF files on disk.
* **Key Classes & Methods:**
  * `PDFLoadError(Exception)`: Custom exception raised when a PDF cannot be parsed or opened.
  * `PDFLoader`: Main loader class.
    * `load_document(file_path: Union[str, Path], metadata: Optional[DocumentMetadata] = None) -> DocumentIngestionResult`: Opens PDF via `fitz.open(str(path))`. Iterates through pages, executes `page.get_text("text")`, strips extraneous control characters, normalizes Unicode whitespace, computes `char_count` and `word_count`, and wraps each page in `DocumentPage`.
* **Error Handling:** Wraps PyMuPDF C-level errors in `try...except fitz.FileDataError` and generic `Exception`, returning a structured `DocumentIngestionResult` with `success=False` instead of crashing the process.
* **Dependencies:** `pymupdf` (`fitz`), `src.ingestion.models`, `src.ingestion.pagination`.

### 1.3 `src/ingestion/pagination.py`
* **Responsibility:** Implements `PaginationResolver` to infer printed page numbers from headers, footers, or offset configurations.
* **Key Functions:**
  * `resolve_printed_page(physical_page: int, page_offset: int = 0) -> int`: Simple offset arithmetic.
  * `PaginationResolver.infer_page_from_text(page_text: str, physical_page: int) -> Optional[int]`: Inspects the first and last two lines of page text using regex patterns (`r'^\s*(\d{1,4})\s*$'`) to detect printed folio numbers, falling back to `physical_page` if ambiguous.

### 1.4 `src/ingestion/registry.py`
* **Responsibility:** Validates and loads canonical metadata from `evaluation/corpus/metadata/paper_registry.json`.
* **Key Functions:**
  * `load_paper_registry(registry_path: Optional[Path] = None) -> Dict[str, DocumentMetadata]`: Parses JSON, validates that each paper has an ID (`P1` to `P5`), title, filename, and page count, returning a keyed dictionary. Raises `RegistryValidationError` on missing or corrupt registry schemas.

---

## 2. Text Chunking Module (`src/chunking/`)

### 2.1 `src/chunking/models.py`
* **Responsibility:** Pydantic models for boundary-aware chunks and chunking configuration.
* **Key Classes:**
  * `ChunkingConfig(BaseModel)`: Configuration parameters: `max_tokens: int = 500`, `overlap_tokens: int = 50`, `min_chunk_tokens: int = 30`. Includes model validator `validate_overlap` ensuring `overlap_tokens < max_tokens`.
  * `ChunkMetadata(BaseModel)`: Fine-grained chunk provenance: `chunk_id: str`, `document_id: str`, `document_title: str`, `physical_page_index: int`, `physical_page_number: int`, `printed_page_number: Optional[int]`, `chunk_index_in_page: int`, `chunk_index_in_doc: int`, `char_count: int`, `word_count: int`, `start_char_idx: int`, `end_char_idx: int`.
  * `DocumentChunk(BaseModel)`: Complete chunk record: `chunk_id: str`, `text: str`, `metadata: ChunkMetadata`.
  * `ChunkingSummary(BaseModel)`: Summary of chunking operations: `total_pages_processed: int`, `total_chunks_created: int`, `average_tokens_per_chunk: float`, `document_id: str`.

### 2.2 `src/chunking/text_chunker.py`
* **Responsibility:** Boundary-aware text segmentation preserving sentences and paragraphs.
* **Key Classes & Methods:**
  * `TextChunker`:
    * `_is_abbreviation_or_initial(text: str, match_pos: int) -> bool`: Regex heuristic inspecting word boundaries preceding a period to avoid splitting on common academic abbreviations (`et al.`, `e.g.`, `i.e.`, `Fig.`, `Tab.`, `Eq.`, `Dr.`, `vs.`, `Vol.`).
    * `segment_text(text: str) -> List[str]`: Splits raw page text into complete sentences using punctuation lookaheads (`(?<=[.!?])\s+(?=[A-Z0-9"\'])`), while preserving abbreviations.
    * `chunk_page(page: DocumentPage, page_index_in_doc: int, start_chunk_idx: int) -> List[DocumentChunk]`: Accumulates sentences into chunks up to `max_tokens`. When a chunk reaches capacity, it preserves the last `overlap_tokens` worth of complete sentences for the next chunk, preventing sentence clipping.
    * `chunk_document_pages(pages: List[DocumentPage]) -> List[DocumentChunk]`: Loops over all pages in a document, assigning deterministic chunk identifiers in the format `{doc_id}_p{page_num:02d}_c{chunk_num:02d}` (e.g., `P1_p03_c02`).
* **Design Decision:** Word count approximation ($1 \text{ token} \approx 0.75 \text{ words}$) is used for rapid, zero-dependency token budget estimation without requiring heavy tokenizer imports during chunking.

---

## 3. Vector Indexing Module (`src/indexing/`)

### 3.1 `src/indexing/embeddings.py`
* **Responsibility:** Dense vector embedding generation using SentenceTransformers.
* **Key Classes:**
  * `BaseEmbedder(ABC)`: Abstract interface declaring `embed_text(text: str) -> List[float]`, `embed_batch(texts: List[str]) -> List[List[float]]`, `embedding_dim -> int`, and `model_name -> str`.
  * `SentenceTransformerEmbedder(BaseEmbedder)`: Wraps `sentence-transformers/all-MiniLM-L6-v2`. Lazily loads the PyTorch model on first invocation to accelerate initial CLI startup. Returns 384-dimensional $L_2$-normalized float vectors.
  * `MockEmbedder(BaseEmbedder)`: Deterministic mock generator producing pseudo-random 384-dimensional unit vectors based on a hash of the input text; used in unit tests to eliminate model download requirements.

### 3.2 `src/indexing/vector_store.py`
* **Responsibility:** Embedded Qdrant vector database lifecycle and point upsert operations.
* **Key Classes & Methods:**
  * `QdrantStoreManager`:
    * `__init__(qdrant_path: Optional[Path] = None, client: Optional[QdrantClient] = None)`: Connects to local persistent Qdrant at `data/qdrant/` using embedded RocksDB storage.
    * `chunk_id_to_point_id(chunk_id: str) -> int`: Generates deterministic, 64-bit integer point IDs for Qdrant using `hashlib.sha256(chunk_id.encode()).digest()[:8]`, ensuring idempotency on re-indexing.
    * `ensure_collection(collection_name: str, vector_dim: int = 384, distance: Distance = Distance.COSINE) -> None`: Creates the collection if absent, configuring HNSW graph indexing and Cosine similarity.
    * `upsert_chunks(collection_name: str, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int`: Batches points into Qdrant `PointStruct` instances containing the dense vector and complete payload metadata (`document_id`, `document_title`, `physical_page_number`, `printed_page_number`, `text`, `chunk_id`).
    * `get_point_count(collection_name: str) -> int`: Returns exact count of indexed vectors in collection.
    * `close() -> None`: Releases RocksDB storage locks cleanly.

### 3.3 `src/indexing/pipeline.py`
* **Responsibility:** Orchestrates end-to-end embedding and vector upsert from chunked JSON files.
* **Key Classes:**
  * `IndexingPipeline`:
    * `load_and_validate_chunks(chunks_path: Path) -> List[DocumentChunk]`: Loads serialized chunks from disk and verifies schema validity.
    * `index_chunks(chunks: List[DocumentChunk], dry_run: bool = False) -> IndexingSummary`: Batches chunks (default batch size 64), computes embeddings via `embedder.embed_batch()`, and upserts vectors into Qdrant. Returns `IndexingSummary` with point counts and duration.

---

## 4. Multi-Modal Retrieval Module (`src/retrieval/`)

### 4.1 `src/retrieval/models.py`
* **Responsibility:** Pydantic models for search requests, retrieved passages, and ranking summaries.
* **Key Classes:**
  * `RetrievedChunk(BaseModel)`: Represents a candidate passage retrieved from either dense, sparse, or hybrid search. Contains `chunk_id`, `document_id`, `document_title`, `physical_page_number`, `printed_page_number`, `text`, `score: float`, `retrieval_method: Optional[str]`, `dense_rank: Optional[int]`, `bm25_rank: Optional[int]`, `dense_score: Optional[float]`, `bm25_score: Optional[float]`.
    * Method `formatted_citation() -> str`: Formats human-readable citation string: `"[P1, Page 3, Chunk P1_p03_c02] Retrieval-Augmented Generation..."`.
  * `RetrievalResult(BaseModel)`: Container for query retrieval operations: `query: str`, `retrieval_mode: str`, `chunks: List[RetrievedChunk]`, `total_candidates: int`, `execution_time_ms: float`.
  * `HybridRetrievalConfig(BaseModel)`: Settings for hybrid search: `dense_top_k: int = 20`, `bm25_top_k: int = 20`, `final_top_k: int = 5`, `rrf_k: int = 60`, `dense_weight: float = 1.0`, `bm25_weight: float = 1.0`, `enable_intent_routing: bool = False`.

### 4.2 `src/retrieval/retriever.py`
* **Responsibility:** Dense vector retrieval executing cosine nearest-neighbor search in Qdrant.
* **Key Classes & Methods:**
  * `DenseRetriever`:
    * `validate_query(query: str) -> str`: Rejects empty or whitespace-only queries by raising `InvalidQueryError`.
    * `build_document_filter(document_ids: Optional[List[str]]) -> Optional[Filter]`: Constructs Qdrant metadata filters (`FieldCondition(key="document_id", match=MatchAny(any=...))`) to restrict search to specific papers.
    * `retrieve(query: str, top_k: int = 5, min_score: Optional[float] = None, document_ids: Optional[List[str]] = None) -> RetrievalResult`: Computes 384-d query embedding via `embedder.embed_text(query)`, calls `client.search()` with cosine distance, maps search hits to `RetrievedChunk`, and applies optional `min_score` cutoff.

### 4.3 `src/retrieval/bm25.py`
* **Responsibility:** Sparse lexical keyword retrieval using the Okapi BM25 ranking function.
* **Key Classes & Methods:**
  * `BM25Index`:
    * `tokenize(text: str) -> List[str]`: Converts text to lowercase, strips punctuation, splits on whitespace, and filters single-character noise tokens.
    * `from_chunks_file(chunks_file: Path) -> BM25Index`: Loads `corpus_chunks.json`, tokenizes every chunk text, and fits `BM25Okapi` inverted index.
    * `search(query: str, top_k: int = 20, document_ids: Optional[List[str]] = None) -> List[RetrievedChunk]`: Tokenizes query, calls `bm25.get_scores()`, applies optional document ID filter masks, normalizes BM25 scores to $[0, 1]$, and returns top-$k$ ranked chunks.
  * `BM25Retriever`: Façade wrapping `BM25Index` conforming to standard retrieval interfaces.

### 4.4 `src/retrieval/fusion.py`
* **Responsibility:** Reciprocal Rank Fusion (RRF) algorithm implementation.
* **Key Functions:**
  * `reciprocal_rank_fusion(dense_chunks: List[RetrievedChunk], bm25_chunks: List[RetrievedChunk], rrf_k: int = 60, dense_weight: float = 1.0, bm25_weight: float = 1.0) -> List[RetrievedChunk]`:
    * Assigns 1-based ranks to dense candidates ($r_d$) and BM25 candidates ($r_b$).
    * Calculates combined fusion score:
      $$\text{Score}(c) = \text{dense\_weight} \cdot \frac{1}{rrf\_k + r_d(c)} + \text{bm25\_weight} \cdot \frac{1}{rrf\_k + r_b(c)}$$
    * Propagates both `dense_rank`, `bm25_rank`, `dense_score`, and `bm25_score` onto the merged chunk instance.
    * Sorts descending by RRF score with deterministic secondary tie-breaking on `chunk_id`.

### 4.5 `src/retrieval/hybrid.py`
* **Responsibility:** Coordinates dense and sparse retrievals, applies intent routing, and fuses candidates.
* **Key Classes & Methods:**
  * `HybridRetriever`:
    * `retrieve(query: str, top_k: Optional[int] = None, min_score: Optional[float] = None, document_ids: Optional[List[str]] = None) -> RetrievalResult`:
      1. If `enable_intent_routing=True`, executes `QueryIntentClassifier` to detect if the query is comparative.
      2. Runs `dense_retriever.retrieve(query, top_k=dense_top_k)`.
      3. Runs `bm25_retriever.retrieve(query, top_k=bm25_top_k)`.
      4. Fuses lists using `reciprocal_rank_fusion()`.
      5. If comparative query routing is active, applies source-capping constraint (e.g., maximum 2 chunks per paper in top-$k$) to guarantee multi-paper evidentiary balance.
      6. Returns top-$k$ fused `RetrievalResult`.

### 4.6 `src/retrieval/router.py`
* **Responsibility:** Rule-based query intent classification for adaptive retrieval routing.
* **Key Classes & Functions:**
  * `QueryIntent(str, Enum)`: `STANDARD = "standard"`, `COMPARATIVE = "comparative"`.
  * `classify_query_intent(query: str) -> QueryIntentResult`:
    * Checks for comparative syntax cues: `compare`, `comparison`, `versus`, `vs`, `difference between`, `differ`, `tradeoff`.
    * Identifies entity mentions across known paper titles and aliases: `RAG`, `DPR`, `REALM`, `Self-RAG`, `Ragas`, `BART`, `BM25`.
    * Flags query as `COMPARATIVE` if multiple distinct paper entities and comparative cues are present, directing `HybridRetriever` to balance candidates across papers.

---

## 5. Evidence-Grounded Generation Module (`src/generation/`)

### 5.1 `src/generation/models.py`
* **Responsibility:** Pydantic models for structured generation output, citations, and answer status.
* **Key Classes:**
  * `AnswerStatus(str, Enum)`:
    * `ANSWERED`: Factual answer substantiated by retrieved passages.
    * `PARTIALLY_ANSWERED`: Evidence covers part of a multi-part query.
    * `ABSTAINED_INSUFFICIENT_EVIDENCE`: Corpus lacks sufficient facts to answer.
    * `ABSTAINED_OUT_OF_SCOPE`: Query falls outside indexed scientific corpus.
    * `CONFLICTING_EVIDENCE`: Retrieved passages contain direct contradictions.
    * `ERROR`: Upstream provider failure, timeout, or schema rejection.
  * `Citation(BaseModel)`: Verified citation structure: `document_id: str`, `document_title: str`, `physical_page_number: int`, `printed_page_number: Optional[int]`, `chunk_id: str`, `excerpt: str`.
  * `GenerationConfig(BaseModel)`: Generation parameters: `provider_type: str = "mock"`, `model_name: Optional[str]`, `temperature: float = 0.0`, `max_tokens: int = 1024`, `max_completion_tokens: Optional[int]`, `top_k: int = 5`, `timeout_seconds: float = 30.0`.
  * `GeneratedAnswer(BaseModel)`: Final response container: `question: str`, `answer: str`, `status: AnswerStatus`, `citations: List[Citation]`, `retrieved_chunks: List[RetrievedChunk]`, `model_name: str`, `provider_name: str`, `retrieval_mode: str`, `latency_ms: float`, `confidence_notes: Optional[str]`, `prompt_tokens: Optional[int]`, `completion_tokens: Optional[int]`, `total_tokens: Optional[int]`.

### 5.2 `src/generation/prompts.py`
* **Responsibility:** Prompt templates enforcing structured JSON output, citation provenance, and delimiter isolation.
* **Key Functions:**
  * `format_evidence_block(chunks: List[RetrievedChunk]) -> str`: Formats retrieved passages into isolated, unambiguous evidence delimiters:
    ```text
    === PASSAGE [P1_p03_c02] (Paper: P1, Page: 3) ===
    Text: The generator component uses BART-large with 400M parameters...
    ```
  * `build_generation_messages(question: str, chunks: List[RetrievedChunk], system_prompt_override: Optional[str] = None) -> List[Dict[str, str]]`:
    * Constructs system prompt instructing model to act as a strict research assistant.
    * Mandates JSON output schema: `{"status": "...", "answer": "...", "citations": [{"chunk_id": "...", "excerpt": "..."}]}`.
    * Strictly prohibits citing chunk IDs not present in context.
    * Instructs model to output `status="abstained_insufficient_evidence"` and empty citations if evidence is absent.

### 5.3 `src/generation/generator.py`
* **Responsibility:** Coordinates retrieval, LLM completion, citation provenance verification, and excerpt reconciliation.
* **Key Classes & Methods:**
  * `AnswerGenerator`:
    * `_validate_and_build_citations(raw_citations: Any, chunks_map: Dict[str, RetrievedChunk]) -> Tuple[List[Citation], List[str]]`:
      1. Enforces `raw_citations` is a list of dictionaries.
      2. Validates `chunk_id` exists in `chunks_map`. **Rejects hallucinated chunk IDs** not present in retrieved context.
      3. Deduplicates citations.
      4. **Excerpt Reconciliation:** Checks if model-emitted excerpt is a verbatim substring of source chunk text. If paraphrased or not found, automatically reconciles by extracting the verbatim passage prefix (`chunk.text[:120] + "..."`), guaranteeing grounded quotes.
      5. Populates canonical metadata (`document_id`, `document_title`, `physical_page_number`) directly from the verified chunk, preventing model fabrication.
    * `_parse_model_output(content: str, chunks_map: Dict[str, RetrievedChunk]) -> Tuple[str, AnswerStatus, List[Citation], Optional[str]]`:
      * Strips markdown fences (````json ... ````).
      * Parses JSON and validates required fields (`status`, `answer`, `citations`).
      * Handles plain-text fallback: detects plain-text abstentions; extracts explicit chunk IDs if present; rejects unstructured arbitrary plain text as `AnswerStatus.ERROR`.
    * `generate(question: str, retrieval_result: Optional[RetrievalResult] = None, top_k: Optional[int] = None, min_score: Optional[float] = None) -> GeneratedAnswer`:
      * Rejects empty queries immediately.
      * Executes retrieval if not supplied.
      * **Immediate Abstention Safeguard:** If retrieval returns 0 chunks, immediately returns `ABSTAINED_INSUFFICIENT_EVIDENCE` without invoking external LLM APIs, saving cost and eliminating hallucination risk.
      * Invokes LLM provider with `timeout` and `response_format="json_object"`.
      * Measures wall-clock latency and token usage.
      * Handles `ProviderAuthError`, `ProviderTimeoutError`, and `RateLimitError` gracefully.

### 5.4 `src/generation/providers/`
* **`base.py`:** Defines `BaseLLMProvider(ABC)` declaring `generate()`, `name`, and `default_model`. Defines standard exception hierarchy: `GenerationError`, `ProviderError`, `ProviderAuthError`, `ProviderTimeoutError`, `RateLimitError`, `MalformedResponseError`.
* **`factory.py`:** `get_llm_provider(provider_type: str, model_name: Optional[str])`:
  * Resolves requested provider (`mock`, `gemini`, `openai`, `groq`, `ollama`, or `auto`).
  * If `auto`, selects first configured provider with a valid API key (Gemini -> OpenAI -> Groq -> Ollama -> Mock).
* **`mock.py`:** `MockLLMProvider`:
  * Provides deterministic offline synthesis for CI and reproducible benchmarking.
  * Scans retrieved passages for keywords matching expected benchmark answers; emits structured JSON with exact chunk ID and verbatim excerpts.
* **`gemini_provider.py`:** Google Gemini adapter using official `google-genai` SDK (`gemini-2.5-flash`). Enforces JSON mode via `response_mime_type="application/json"`.
* **`groq_provider.py`:** Groq Cloud adapter using official `groq` SDK (`llama-3.3-70b-versatile`). Features `groq_generation_response_format()` and `max_completion_tokens` parameter mapping.
* **`openai_provider.py`:** OpenAI adapter using official `openai` SDK (`gpt-4o-mini`, `gpt-4o`). Supports both cloud OpenAI and local Ollama instances via `OLLAMA_BASE_URL` (`http://localhost:11434/v1`).

---

## 6. Shared Service & API Layer (`src/api/` & `src/ui/`)

### 6.1 `src/ui/service.py`
* **Responsibility:** Centralized thread-safe application service façade (`ResearchEngineService`) managing singleton instances of embedders, vector store managers, loaders, and chunkers.
* **Key Methods:**
  * `get_system_status() -> Dict[str, Any]`: Queries Qdrant collection point counts, indexed paper tallies, and audits provider API key configuration without exposing credentials.
  * `get_indexed_papers() -> List[PaperSummary]`: Aggregates indexed documents from Qdrant vectors and local chunk metadata.
  * `ingest_paper(pdf_path: Path, doc_id: Optional[str], doc_title: Optional[str], dry_run: bool = False) -> IngestionReport`:
    1. Loads PDF via `PDFLoader`.
    2. Chunks pages via `TextChunker`.
    3. If `dry_run=False`, embeds and upserts vectors into Qdrant via `IndexingPipeline`.
    4. Synchronizes new chunks into `data/chunked/corpus_chunks.json` for BM25 with automated `.bak` backup.
  * `ask_question(...) -> GeneratedAnswer`: Builds `AnswerGenerator` sharing pre-loaded embedders, executes hybrid retrieval with optional intent routing, and returns cited answer.

### 6.2 `src/api/app.py` & `src/api/dependencies.py`
* **Responsibility:** FastAPI application factory with lifespan management and dependency injection.
* **Lifespan Management:** In `dependencies.py`, `initialize_service()` initializes the singleton `ResearchEngineService` on FastAPI startup; `shutdown_service()` closes Qdrant RocksDB locks on server shutdown.
* **CORS:** Configures `CORSMiddleware` allowing `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` for smooth local Next.js frontend integration.
* **Router Mounting:** Mounts `/api/system`, `/api/papers`, and `/api/research`. Exposes root `/health` alias for container health checks.

