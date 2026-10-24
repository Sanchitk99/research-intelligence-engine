# 01 — Complete Folder and File Guide

## 1. Repository Directory Structure

```text
research-intelligence-engine/
├── app.py                                  # Root entrypoint for Streamlit application
├── server.py                               # Root entrypoint for FastAPI backend server
├── requirements.txt                        # Python dependencies and version constraints
├── README.md                               # Primary portfolio documentation and project overview
├── .env                                    # Environment variable configuration (API keys, provider settings)
├── .gitignore                              # Git exclusion rules for artifacts, caches, and models
│
├── data/                                   # Local storage for indexes, chunks, and vector database
│   ├── chunked/                            # Serialized text chunks
│   │   ├── corpus_chunks.json              # 396 boundary-aware chunks across 5 benchmark papers
│   │   └── corpus_chunks.json.bak          # Automatic backup created during dynamic paper ingestion
│   ├── ingested/                           # Intermediate page-level extraction records
│   │   └── corpus_pages.json               # Extracted pages with physical page indices
│   └── qdrant/                             # Embedded Qdrant persistent storage
│       ├── .lock                           # RocksDB concurrency lock
│       ├── meta.json                       # Vector collection schema and HNSW indexing metadata
│       └── collection/
│           └── research_corpus/
│               └── storage.sqlite          # Vector payloads and SQLite metadata store
│
├── docs/                                   # Documentation and visual demo assets
│   └── assets/                             # Portfolio screenshots for README and documentation
│       ├── 01_workspace.png                # Research workspace initial state
│       ├── 02_grounded_answer.png          # Synthesized answer with citations and inspector
│       ├── 03_abstention.png               # Out-of-scope query abstention safeguard
│       ├── 04_paper_library.png            # Indexed paper catalog and ingestion modal
│       └── 05_system_status.png            # Telemetry, provider health, and collection statistics
│
├── evaluation/                             # Evaluation benchmark corpus, questions, and reports
│   ├── corpus/                             # Evaluation literature and metadata
│   │   ├── metadata/
│   │   │   └── paper_registry.json         # Authoritative metadata for P1 to P5
│   │   └── papers/                         # Landmark RAG PDF documents
│   │       ├── P1_RAG_Lewis_2020.pdf       # Retrieval-Augmented Generation (Lewis et al., 2020)
│   │       ├── P2_DPR_Karpukhin_2020.pdf   # Dense Passage Retrieval (Karpukhin et al., 2020)
│   │       ├── P3_REALM_Guu_2020.pdf       # REALM Pre-training (Guu et al., 2020)
│   │       ├── P4_SelfRAG_Asai_2023.pdf    # Self-RAG (Asai et al., 2023)
│   │       └── P5_RAGAS_Es_2023.pdf        # Ragas Evaluation (Es et al., 2023)
│   ├── questions/                          # Ground-truth evaluation query datasets
│   │   ├── answerable_questions.json       # 18 ground-truth questions with target papers & pages
│   │   ├── out_of_scope_questions.json     # 10 out-of-scope questions testing abstention
│   │   └── disagreement_pairs.json         # Comparative queries spanning multiple papers
│   └── results/                            # Benchmark reports and raw evaluation JSON records
│       ├── answer_quality_error_analysis.json / .md
│       ├── answer_quality_latency_evaluation.json / .md
│       ├── cross_paper_evaluation.json / .md
│       ├── diversified_retrieval_cross_paper.json / .md
│       ├── hybrid_retrieval_evaluation.json / .md
│       ├── hybrid_vs_dense_comparison_report.md
│       ├── query_intent_routing_report.md
│       └── retrieval_evaluation_report.md
│
├── frontend/                               # Modern Next.js 15 Web Application
│   ├── package.json                        # Frontend dependencies (React 19, Lucide, Tailwind)
│   ├── tsconfig.json                       # TypeScript compiler configuration
│   ├── next.config.ts                      # Next.js configuration
│   ├── public/                             # Static web assets
│   └── src/
│       ├── app/
│       │   ├── globals.css                 # Tailwind CSS directives and custom styling
│       │   ├── layout.tsx                  # Root layout wrapper with Geist font definitions
│       │   └── page.tsx                    # Main multi-view application dashboard (437 lines)
│       ├── components/
│       │   ├── about/                      # About This Project documentation tab
│       │   │   └── ProjectOverview.tsx     # Architecture breakdown, retrieval metrics, reports
│       │   ├── layout/                     # Shell navigation and headers
│       │   │   ├── Header.tsx              # Application top bar with session title & settings
│       │   │   ├── Sidebar.tsx             # Collapsible navigation, session list, clear history
│       │   │   └── SettingsModal.tsx       # Model provider, retrieval mode, and top-k dialog
│       │   ├── papers/                     # Paper catalog and ingestion interfaces
│       │   │   ├── CorpusList.tsx          # Benchmark PDF cards with one-click ingestion
│       │   │   ├── IngestModal.tsx         # Drag-and-drop PDF uploader with dry-run toggle
│       │   │   └── PaperTable.tsx          # Tabular overview of indexed documents and chunks
│       │   ├── research/                   # Evidence-grounded research workspace
│       │   │   ├── AnswerCard.tsx          # Synthesized response, status badge, latency display
│       │   │   ├── CitationCard.tsx        # Interactive page citation with verbatim excerpt
│       │   │   ├── ContextInspector.tsx    # Slide-out drawer to audit all retrieved passages
│       │   │   ├── EmptyWorkspace.tsx      # Welcome state with curated benchmark prompt chips
│       │   │   ├── LoadingSkeleton.tsx     # Animated shimmer during retrieval and generation
│       │   │   └── QueryComposer.tsx       # Search bar with provider selector and submit button
│       │   └── system/                     # Telemetry and engine observability
│       │       ├── ArchitectureCard.tsx    # Pipeline summary and architectural reference
│       │       ├── ProviderGrid.tsx        # Real-time health status of LLM providers
│       │       └── SystemOverview.tsx      # Vector collection metrics and document statistics
│       └── lib/
│           ├── api.ts                      # Client-side API fetch wrapper for FastAPI endpoints
│           ├── types.ts                    # TypeScript data models mirroring Pydantic schemas
│           └── utils.ts                    # Styling helpers (`cn` clsx wrapper, formatting)
│
├── scripts/                                # Operational pipeline entrypoints and evaluation runners
│   ├── run_answer_quality_evaluation.py    # 5-dimensional benchmark runner (latency, citations, abstention)
│   ├── run_chunking.py                     # CLI entrypoint for corpus text chunking
│   ├── run_cross_encoder_reranking_evaluation.py
│   ├── run_cross_paper_evaluation.py       # Benchmark runner for cross-paper comparative queries
│   ├── run_demo.py                         # Interactive terminal demonstration of RAG pipeline
│   ├── run_dense_hybrid_retrieval_comparison.py
│   ├── run_dense_retrieval_diagnostic.py   # Diagnostic analysis of dense retrieval failures
│   ├── run_diversified_retrieval_evaluation.py
│   ├── run_evaluation.py                   # High-level retrieval evaluation entrypoint
│   ├── run_generation_evaluation.py        # Generation evaluation runner across providers
│   ├── run_groq_generation_evaluation.py   # Dedicated Groq evaluation runner
│   ├── run_indexing.py                     # CLI entrypoint for embedding and Qdrant indexing
│   ├── run_ingestion.py                    # CLI entrypoint for PDF text extraction
│   ├── run_live_gemini_generation_evaluation.py
│   ├── run_ollama_generation_evaluation.py # Dedicated Ollama evaluation runner
│   ├── run_qa.py                           # CLI entrypoint for ad-hoc question answering
│   ├── run_query_intent_routing_evaluation.py # Evaluates standard vs. comparative routing
│   ├── run_retrieval.py                    # CLI entrypoint for standalone retrieval queries
│   └── run_smoke_test.py                   # System smoke test validating providers & end-to-end flow
│
├── src/                                    # Core application source code
│   ├── api/                                # FastAPI backend implementation
│   │   ├── app.py                          # FastAPI application factory, CORS, and router registration
│   │   ├── dependencies.py                 # Dependency injection for ResearchEngineService
│   │   ├── schemas.py                      # Pydantic v2 request and response contracts
│   │   └── routes/                         # API endpoint route modules
│   │       ├── papers.py                   # Document catalog, corpus PDF listing, and ingestion
│   │       ├── research.py                 # Question answering endpoint (`/api/research/ask`)
│   │       └── system.py                   # System status and health probes (`/api/system/status`)
│   ├── chunking/                           # Boundary-aware text segmentation
│   │   ├── cli.py                          # Command-line interface for chunking pipeline
│   │   ├── models.py                       # ChunkMetadata, DocumentChunk, ChunkingConfig models
│   │   ├── pipeline.py                     # Orchestration from extracted pages to serialized chunks
│   │   └── text_chunker.py                 # Sentence regex segmentation and token count manager
│   ├── evaluation/                         # Evaluation metrics, dataset loaders, and benchmark logic
│   │   ├── cli.py                          # CLI for running retrieval evaluation suites
│   │   ├── cross_paper_evaluator.py        # Evaluator for multi-paper synthesis questions
│   │   ├── diversified_retrieval_evaluator.py # Evaluates source paper balance and coverage
│   │   ├── evaluator.py                    # RetrievalEvaluator (Hit@k, MRR@k, NDCG@k, Precision@k)
│   │   ├── generation_evaluator.py         # GenerationEvaluator (lexical recall, citation validity)
│   │   ├── groq_generation_runner.py       # Execution harness with Groq rate-limit backoff
│   │   ├── ground_truth.py                 # Loads and indexes ground-truth evaluation questions
│   │   ├── models.py                       # Pydantic models for evaluation metrics and query targets
│   │   ├── ollama_generation_runner.py     # Execution harness for local Ollama models
│   │   └── reporting.py                    # Markdown and JSON report formatting utilities
│   ├── generation/                         # Evidence-grounded answer synthesis and validation
│   │   ├── cli.py                          # Interactive CLI for answer generation
│   │   ├── generator.py                    # AnswerGenerator: prompt assembly, validation, abstention
│   │   ├── models.py                       # GeneratedAnswer, Citation, AnswerStatus, GenerationConfig
│   │   ├── prompts.py                      # Prompt engineering: evidence delimiters and JSON schemas
│   │   └── providers/                      # Pluggable LLM provider implementations
│   │       ├── base.py                     # BaseLLMProvider abstract interface and custom exceptions
│   │       ├── factory.py                  # get_llm_provider factory resolver
│   │       ├── gemini_provider.py          # Google Gemini API integration (`gemini-2.5-flash`)
│   │       ├── groq_provider.py            # Groq Cloud API integration (`llama-3.3-70b-versatile`)
│   │       ├── mock.py                     # Deterministic offline mock provider with keyword extraction
│   │       └── openai_provider.py          # OpenAI API integration (`gpt-4o-mini`, `gpt-4o`)
│   ├── indexing/                           # Dense embedding generation and vector database management
│   │   ├── cli.py                          # CLI interface for vector store indexing
│   │   ├── embeddings.py                   # SentenceTransformerEmbedder (`all-MiniLM-L6-v2`)
│   │   ├── models.py                       # IndexingConfig, IndexingSummary, error classes
│   │   ├── pipeline.py                     # IndexingPipeline: chunk loading, batching, and upsert
│   │   └── vector_store.py                 # QdrantStoreManager: collection lifecycle and point upsert
│   ├── ingestion/                          # Document extraction and pagination resolution
│   │   ├── batch.py                        # Batch ingestion utility across corpus directories
│   │   ├── cli.py                          # CLI interface for PDF ingestion
│   │   ├── loader.py                       # PDFLoader: PyMuPDF text and metadata extraction
│   │   ├── models.py                       # DocumentMetadata, PageMetadata, DocumentPage models
│   │   ├── pagination.py                   # PaginationResolver: maps physical to printed page indices
│   │   └── registry.py                     # Registry loader and validator (`paper_registry.json`)
│   ├── retrieval/                          # Multi-modal search, hybrid fusion, and intent routing
│   │   ├── bm25.py                         # Okapi BM25 index, custom tokenizer, and keyword search
│   │   ├── cli.py                          # CLI interface for executing retrieval queries
│   │   ├── fusion.py                       # Reciprocal Rank Fusion (RRF $k=60$) algorithm
│   │   ├── hybrid.py                       # HybridRetriever combining Dense + BM25 via RRF
│   │   ├── models.py                       # RetrievedChunk, RetrievalResult, HybridRetrievalConfig
│   │   ├── reranker.py                     # CpuCrossEncoderReranker with RAM guardrails
│   │   ├── retriever.py                    # DenseRetriever: cosine similarity search in Qdrant
│   │   └── router.py                       # QueryIntentClassifier: comparative vs. standard intent
│   └── ui/                                 # Shared business service and Streamlit dashboard
│       ├── app.py                          # Streamlit dashboard implementation (700+ lines)
│       └── service.py                      # ResearchEngineService: central façade for API & UI
│
└── tests/                                  # Comprehensive automated test suite (20 test modules)
    ├── conftest.py                         # Pytest fixtures: mock providers, sample chunks, temp dirs
    ├── test_answer_quality_evaluation.py   # Benchmark tests: dataset integrity, provenance, citations
    ├── test_api_endpoints.py               # Unit tests for FastAPI HTTP routes using TestClient
    ├── test_api_live_integration.py        # Integration tests verifying live server request/response
    ├── test_chunking.py                    # Sentence boundary tests, token limits, overlap math
    ├── test_demo_workflow.py               # End-to-end integration test of the demo workflow
    ├── test_diversified_retrieval.py       # Cross-paper retrieval coverage and diversity tests
    ├── test_evaluation.py                  # Hit@k, MRR@k, NDCG@k metric calculation tests
    ├── test_generation.py                  # AnswerGenerator tests, prompt formatting, JSON parsing
    ├── test_groq_generation_runner.py      # Groq runner integration and error handling tests
    ├── test_groq_json_schema_handling.py   # Groq structured output schema compliance tests
    ├── test_groq_rate_limit_handling.py    # Exponential backoff and retry behavior tests
    ├── test_hybrid_retrieval.py            # RRF fusion math, score monotonicity, tie-breaking tests
    ├── test_indexing.py                    # Embedding dimensions, Qdrant collection upsert tests
    ├── test_ingestion.py                   # PyMuPDF loading, physical page tracking, registry tests
    ├── test_ollama_generation_runner.py    # Local Ollama runner and model validation tests
    ├── test_query_intent_routing.py        # Intent classification and comparative capping tests
    ├── test_reranker.py                    # Cross-encoder reranker initialization & RAM check tests
    ├── test_retrieval.py                   # Dense retrieval cosine similarity and filtering tests
    └── test_ui_app.py                      # Streamlit UI service integration tests
```

---

## 2. Directory and File Responsibility Matrix

### 2.1 Root Application Entrypoints & Manifests

| File | Primary Responsibility | Calling Dependencies | Key Outputs / Side Effects |
|---|---|---|---|
| `app.py` | Standalone root entrypoint for launching the Streamlit dashboard (`streamlit run app.py`). Configures `sys.path` and delegates execution to `src/ui/app.py`. | User via shell | Launches interactive Streamlit web server on port 8501. |
| `server.py` | Server entrypoint for the FastAPI REST backend. Sets OpenMP duplicate runtime mitigation (`KMP_DUPLICATE_LIB_OK=TRUE`) and boots Uvicorn on `127.0.0.1:8000`. | User via shell | Spawns FastAPI HTTP server hosting `/api/*` endpoints. |
| `requirements.txt` | Python dependency manifest specifying minimum versions for deep learning (`torch`, `sentence-transformers`), storage (`qdrant-client`, `rank-bm25`, `portalocker`), API (`fastapi`, `uvicorn`), and testing. | `pip install -r requirements.txt` | Defines reproducible Python runtime environment. |
| `.env` | Local environment configuration storing provider API keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`) and Ollama endpoint settings (`OLLAMA_BASE_URL`). | `python-dotenv`, `factory.py`, `service.py` | Supplies credentials without committing secrets to version control. |
| `.gitignore` | Prevents local databases (`data/qdrant/`), temporary test directories (`.pytest-full-temp`), node modules (`node_modules`), build artifacts (`.next`), and virtual environments from entering Git. | Git | Enforces clean repository history. |
| `README.md` | Primary portfolio documentation explaining project purpose, architecture, retrieval benchmarks, interface screenshots, and quickstart instructions. | Technical recruiters, engineers | Communicates project technical credibility and evidence. |

---

### 2.2 Core Source Modules (`src/`)

#### `src/ingestion/` — Document Extraction & Pagination
* **Responsibility:** Ingest raw research paper PDFs, extract clean page text, and resolve physical page numbering for precise grounding.
* **Key Files:**
  * `loader.py`: Uses PyMuPDF (`fitz`) to open PDFs, extract per-page text, clean whitespace, and attach page metadata.
  * `pagination.py`: Defines `PaginationResolver` to map 0-based PDF array indices to 1-based physical page numbers and match printed roman/arabic numerals.
  * `models.py`: Defines Pydantic data models: `DocumentMetadata`, `PageMetadata`, `DocumentPage`, `DocumentIngestionResult`.
  * `registry.py`: Validates and loads `evaluation/corpus/metadata/paper_registry.json` to enforce canonical paper IDs (P1 to P5).
  * `batch.py`: Provides `batch_ingest()` utility to process entire directories of PDFs sequentially.
  * `cli.py`: Command-line interface (`python -m src.ingestion.cli`) to run extraction on demand.

#### `src/chunking/` — Boundary-Aware Text Segmentation
* **Responsibility:** Transform raw document pages into structurally coherent chunks without splitting sentences or severing semantic context.
* **Key Files:**
  * `text_chunker.py`: Implements regex-driven sentence boundary detection (`re.split` with abbreviation lookbehinds for `e.g.`, `i.e.`, `al.`, `Fig.`, `Tab.`). Groups sentences into chunks up to `max_tokens` (default 500) with configurable overlap (default 50 tokens).
  * `models.py`: Defines `ChunkingConfig`, `ChunkMetadata` (tracking physical page numbers, word count, character indices), `DocumentChunk`, and `ChunkingSummary`.
  * `pipeline.py`: Orchestrates multi-page chunking across a corpus, assigning deterministic IDs (e.g., `P1_p03_c02`).
  * `cli.py`: Command-line tool to run chunking on extracted pages.

#### `src/indexing/` — Vector Embeddings & Persistent Storage
* **Responsibility:** Convert text chunks into dense semantic vectors and store them in a persistent local vector database.
* **Key Files:**
  * `embeddings.py`: Wraps `SentenceTransformer("all-MiniLM-L6-v2")` to produce 384-dimensional dense vectors; provides `MockEmbedder` for fast unit tests.
  * `vector_store.py`: `QdrantStoreManager` initializes an embedded Qdrant instance storing vectors in `data/qdrant/` with RocksDB backend and Cosine distance metric.
  * `models.py`: Defines `IndexingConfig`, `IndexingSummary`, and custom error hierarchies (`VectorStoreError`, `IncompatibleCollectionError`).
  * `pipeline.py`: Coordinates chunk loading, batch embedding computation, and vector upsert into Qdrant.
  * `cli.py`: CLI entrypoint to index serialized chunks.

#### `src/retrieval/` — Multi-Modal Search & Hybrid Fusion
* **Responsibility:** Retrieve the most relevant passages for a question using dense semantic matching, sparse lexical search, or combined hybrid ranking.
* **Key Files:**
  * `retriever.py`: `DenseRetriever` embeds user queries and executes nearest-neighbor cosine search in Qdrant with optional document filtering.
  * `bm25.py`: `BM25Index` and `BM25Retriever` tokenize queries and rank chunks using Okapi BM25 ($k_1=1.5, b=0.75$) loaded from `corpus_chunks.json`.
  * `fusion.py`: Implements **Reciprocal Rank Fusion (RRF)**: $\text{RRF\_Score}(d) = \sum_{m} \frac{1}{k + \text{rank}_m(d)}$ with smoothing constant $k=60$.
  * `hybrid.py`: `HybridRetriever` runs dense and BM25 retrievals concurrently, fuses ranked lists via RRF, and formats top-$k$ candidates with full provenance.
  * `router.py`: `QueryIntentClassifier` inspects query syntax to detect comparative multi-entity questions (e.g., "compare RAG and DPR on Natural Questions") and enforce per-paper candidate caps.
  * `reranker.py`: `CpuCrossEncoderReranker` provides experimental second-stage cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) guarded by available RAM checks.
  * `models.py`: Defines `RetrievedChunk`, `RetrievalResult`, `RetrievalConfig`, `BM25Config`, `HybridRetrievalConfig`.

#### `src/generation/` — Evidence-Grounded Synthesis & Provenance Validation
* **Responsibility:** Synthesize answers from retrieved passages, enforce structured JSON schemas, validate every emitted citation, and safeguard against hallucination.
* **Key Files:**
  * `generator.py`: `AnswerGenerator` coordinates retrieval, prompt construction, LLM invocation, citation verification, and excerpt reconciliation. Rejects hallucinated chunk IDs and enforces explicit abstention.
  * `prompts.py`: Builds system and user prompts with explicit delimiter blocks (`=== PASSAGE [chunk_id] (Paper, Page X) ===`) and strict JSON schema instructions.
  * `models.py`: Defines `AnswerStatus` enum (`answered`, `partially_answered`, `abstained_insufficient_evidence`, `abstained_out_of_scope`, `conflicting_evidence`, `error`), `Citation`, `GenerationConfig`, `GeneratedAnswer`.
  * `providers/base.py`: Defines `BaseLLMProvider` abstract base class and standardized exceptions (`ProviderAuthError`, `ProviderTimeoutError`, `RateLimitError`).
  * `providers/factory.py`: `get_llm_provider()` factory resolving providers dynamically based on availability and request.
  * `providers/mock.py`: `MockLLMProvider` generates deterministic, context-grounded responses by isolating keywords in retrieved passages; used for reproducible offline testing without API costs.
  * `providers/gemini_provider.py`, `groq_provider.py`, `openai_provider.py`: Production LLM provider adapters with JSON response formatting and token cap controls.

#### `src/api/` & `src/ui/` — Services & Web Interfaces
* **Responsibility:** Expose research capabilities through modern REST endpoints, stateful service facades, and responsive graphical interfaces.
* **Key Files:**
  * `src/ui/service.py`: `ResearchEngineService` serves as the centralized singleton façade, maintaining shared in-memory references to `embedder`, `store_manager`, `loader`, and `chunker`, providing thread-safe question answering, system telemetry, and dynamic PDF ingestion.
  * `src/api/app.py`: Configures FastAPI application, sets permissive development CORS, handles exception middleware, and mounts routers.
  * `src/api/dependencies.py`: FastAPI dependency injection provider supplying `ResearchEngineService` singletons to route handlers.
  * `src/api/schemas.py`: Pydantic request and response schemas: `AskQuestionRequest`, `AskQuestionResponse`, `SystemStatusResponse`, `PaperSummaryResponse`, `IngestPaperRequest`, `IngestionResponse`.
  * `src/api/routes/research.py`: Exposes `POST /api/research/ask` for end-to-end question answering.
  * `src/api/routes/papers.py`: Exposes `GET /api/papers`, `GET /api/papers/corpus`, `POST /api/papers/ingest`, and `POST /api/papers/upload`.
  * `src/api/routes/system.py`: Exposes `GET /api/system/health` and `GET /api/system/status`.
  * `src/ui/app.py`: Comprehensive Streamlit application providing interactive research workspaces, paper library tables, and system diagnostics.

---

### 2.3 Next.js Frontend Modules (`frontend/src/`)

| File / Component | Primary Responsibility | Calling Dependencies | Key User Interactions Handled |
|---|---|---|---|
| `app/page.tsx` | Main application shell controlling view state (`research`, `papers`, `status`, `about`), session history, and question submission. | Next.js App Router | Orchestrates top-level navigation, localStorage persistence, and error banners. |
| `components/research/QueryComposer.tsx` | Search input bar with quick provider selector, retrieval mode toggle, and submit button. | `page.tsx` | Captures user queries, triggers `askQuestion()` API, handles Enter key submission. |
| `components/research/AnswerCard.tsx` | Displays synthesized answer, status badge, latency timer, confidence notes, and citation pills. | `page.tsx` | Renders grounded answers, formats markdown responses, displays explicit abstention alerts. |
| `components/research/CitationCard.tsx` | Card displaying paper title, physical page number, chunk ID, and verbatim source excerpt. | `AnswerCard.tsx` | Highlights citation grounding; allows user to inspect exact passage cited by the LLM. |
| `components/research/ContextInspector.tsx` | Slide-out drawer or expandable panel listing all top-$k$ retrieved chunks with similarity scores. | `AnswerCard.tsx` | Allows complete human audit of all evidence passages provided to the model. |
| `components/research/EmptyWorkspace.tsx` | Curated starting screen displaying benchmark query cards (factual, comparative, out-of-scope). | `page.tsx` | Provides one-click prompt injection to test engine capabilities immediately. |
| `components/research/LoadingSkeleton.tsx` | Animated pulse placeholder displayed while awaiting backend retrieval and synthesis. | `page.tsx` | Provides visual feedback during multi-step pipeline execution. |
| `components/papers/PaperTable.tsx` | Tabular overview of indexed corpus papers showing paper IDs, titles, page counts, and chunks. | `page.tsx` (Papers tab) | Displays live collection state queried from `/api/papers`. |
| `components/papers/CorpusList.tsx` | Grid of benchmark PDF documents available on disk with instant one-click ingestion buttons. | `page.tsx` (Papers tab) | Triggers `/api/papers/ingest` to re-index benchmark literature dynamically. |
| `components/papers/IngestModal.tsx` | Drag-and-drop modal dialog supporting custom PDF uploads with dry-run validation toggle. | `page.tsx` | Calls `uploadAndIngestPDF()` via multipart form upload to ingest custom literature. |
| `components/system/SystemOverview.tsx` | High-level telemetry cards displaying collection name, total vector points, and indexed papers. | `page.tsx` (Status tab) | Observability dashboard fed by `/api/system/status`. |
| `components/system/ProviderGrid.tsx` | Grid displaying configured status and default models for Gemini, OpenAI, Groq, and Ollama. | `page.tsx` (Status tab) | Shows which LLM providers have valid API keys without exposing secrets. |
| `components/system/ArchitectureCard.tsx` | Visual summary of RIE's 4-stage pipeline, design principles, and failure modes governed. | `page.tsx` (Status tab) | Educates visitors on the technical architecture. |
| `components/about/ProjectOverview.tsx` | Comprehensive project overview page detailing architecture, retrieval metrics, and limitations. | `page.tsx` (About tab) | Technical showcase for recruiters and peer reviewers. |
| `components/layout/Header.tsx` | Top header bar showing active session title, status indicator, and settings trigger. | `page.tsx` | Allows opening settings modal and checking API reachability. |
| `components/layout/Sidebar.tsx` | Collapsible sidebar with navigation tabs, session history list, and "New Session" button. | `page.tsx` | Manages multi-turn research sessions saved in localStorage. |
| `components/layout/SettingsModal.tsx` | Modal dialog allowing users to customize active provider, model name, retrieval mode, and top-$k$. | `page.tsx` | Updates engine parameters persisted in localStorage. |
| `lib/api.ts` | Type-safe fetch client for all backend REST endpoints with unified error handling. | All components | Connects Next.js frontend to FastAPI backend (`http://localhost:8000`). |
| `lib/types.ts` | TypeScript interface definitions mirroring FastAPI backend Pydantic schemas. | All components | Guarantees type safety across frontend and backend boundaries. |

---

### 2.4 Test Suite Catalog (`tests/`)

| Test File | Total Tests | Primary Verification Scope |
|---|:---:|---|
| `tests/test_answer_quality_evaluation.py` | 7 | Dataset integrity, citation validity helper, abstention evaluation, unsupported risk formula, claim support keyword heuristic, Ollama detection, percentile math. |
| `tests/test_api_endpoints.py` | 11 | FastAPI route contracts via `TestClient`: health, status, paper listing, corpus PDFs, dry-run ingestion, research ask validation, empty query rejection. |
| `tests/test_api_live_integration.py` | 4 | Live server integration: end-to-end question answering, citation schema compliance, provider status structure. |
| `tests/test_chunking.py` | 16 | Text chunker sentence boundary preservation, abbreviation handling (`e.g.`, `i.e.`), token limits, overlap math, metadata propagation. |
| `tests/test_demo_workflow.py` | 3 | End-to-end execution of `scripts/run_demo.py` workflow with mock providers and hybrid retrieval. |
| `tests/test_diversified_retrieval.py` | 8 | Multi-paper comparative retrieval, paper coverage balance, source-capping constraints. |
| `tests/test_evaluation.py` | 18 | Hit@k, MRR@k, NDCG@k, Precision@k retrieval metric calculations against ground-truth targets. |
| `tests/test_generation.py` | 84 | AnswerGenerator prompt formatting, JSON response parsing, markdown stripping, citation validation, excerpt reconciliation, provider error propagation, real corpus Qdrant integration. |
| `tests/test_groq_generation_runner.py` | 6 | Groq evaluation runner batching, checkpointing, and model invocation. |
| `tests/test_groq_json_schema_handling.py` | 5 | Groq `response_format={"type": "json_object"}` payload structuring and model parameter mapping. |
| `tests/test_groq_rate_limit_handling.py` | 4 | HTTP 429 RateLimitError handling, exponential backoff, and retry delay calculation. |
| `tests/test_hybrid_retrieval.py` | 14 | Reciprocal Rank Fusion ($k=60$) mathematical properties, rank monotonicity, tie breaking, score normalization, sparse-dense agreement. |
| `tests/test_indexing.py` | 12 | SentenceTransformer embedding dimensions (384), Qdrant collection creation, payload storage, point upsert, point count queries. |
| `tests/test_ingestion.py` | 10 | PyMuPDF PDF loading, physical page tracking, whitespace normalization, paper registry validation. |
| `tests/test_ollama_generation_runner.py` | 5 | Local Ollama endpoint reachability, model tag querying, generation runner execution. |
| `tests/test_query_intent_routing.py` | 32 | Query intent classifier regexes, comparative cue extraction, entity recognition, source-capping behavior. |
| `tests/test_reranker.py` | 4 | Cross-encoder model initialization, RAM availability checks, candidate deduplication. |
| `tests/test_retrieval.py` | 12 | DenseRetriever cosine search, top-$k$ slicing, score threshold filtering, empty query handling. |
| `tests/test_ui_app.py` | 8 | ResearchEngineService lifecycle, Streamlit state initialization, dynamic ingestion report generation. |
| `tests/conftest.py` | Fixtures | Common pytest fixtures: mock embedder, mock vector store, sample chunks, temporary directories. |

