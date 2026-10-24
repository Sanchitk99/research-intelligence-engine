# 11 — Complete File Coverage & Audit Index

## Executive Summary

This document serves as the formal audit ledger and verification manifest for the entire **Research Intelligence Engine (RIE)** documentation suite. It inventories all 80+ files across the repository, maps them to their corresponding chapters in `project_documentation/`, details explicit operational exclusions, and provides a completed verification checklist.

---

## 1. Complete File-by-File Coverage Matrix

### 1.1 Root Configuration, Entrypoints & Manifests

| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `README.md` | Primary GitHub project overview, quickstart, and metrics summary | `00_PROJECT_OVERVIEW.md`, `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `requirements.txt` | Python runtime dependencies specification | `07_SECURITY_AND_CONFIGURATION.md`, `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `server.py` | FastAPI application launcher and Uvicorn entrypoint | `02_CODEBASE_DEEP_DIVE.md`, `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `app.py` | Streamlit standalone application launcher | `02_CODEBASE_DEEP_DIVE.md`, `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `.gitignore` | Version control exclusion patterns | `01_FOLDER_AND_FILE_GUIDE.md`, `11_FILE_COVERAGE_INDEX.md` |

---

### 1.2 Core Source Backend Modules (`src/`)

#### Ingestion Subsystem (`src/ingestion/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/ingestion/__init__.py` | Module exports for ingestion models and components | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/ingestion/loader.py` | PDF parsing via PyMuPDF (`fitz`), text extraction | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/ingestion/pagination.py` | Page offset resolution & physical/printed number mapping | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/ingestion/registry.py` | Paper metadata registry loader & catalog validator | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/ingestion/batch.py` | Batch processing orchestrator for PDF corpus extraction | `02_CODEBASE_DEEP_DIVE.md`, `03_ARCHITECTURE_AND_FLOWS.md` |
| `src/ingestion/models.py` | Pydantic data schemas: `DocumentMetadata`, `DocumentPage` | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/ingestion/cli.py` | Command-line interface for PDF batch ingestion | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

#### Chunking Subsystem (`src/chunking/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/chunking/__init__.py` | Module exports for chunking models and pipeline | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/chunking/text_chunker.py` | Sentence-boundary chunker with abbreviation protection | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/chunking/pipeline.py` | Chunking orchestrator converting pages into corpus chunks | `02_CODEBASE_DEEP_DIVE.md`, `03_ARCHITECTURE_AND_FLOWS.md` |
| `src/chunking/models.py` | Pydantic schemas: `DocumentChunk`, `ChunkMetadata` | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/chunking/cli.py` | Command-line interface for corpus text chunking | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

#### Indexing Subsystem (`src/indexing/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/indexing/__init__.py` | Module exports for vector store and embeddings | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/indexing/embeddings.py` | Dense vector embedding generation via `all-MiniLM-L6-v2` | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/indexing/vector_store.py` | Local embedded Qdrant (RocksDB) manager and UUIDv5 IDs | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/indexing/pipeline.py` | Indexing coordinator batching embeddings and upserts | `02_CODEBASE_DEEP_DIVE.md`, `03_ARCHITECTURE_AND_FLOWS.md` |
| `src/indexing/models.py` | Configuration models: `IndexingConfig`, `IndexingSummary` | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/indexing/cli.py` | Command-line interface for vector store indexing | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

#### Retrieval Subsystem (`src/retrieval/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/retrieval/__init__.py` | Module exports for dense, BM25, and hybrid retrievers | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/retrieval/retriever.py` | Dense semantic vector retriever querying local Qdrant | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/retrieval/bm25.py` | Okapi BM25 keyword searcher on `corpus_chunks.json` | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/retrieval/fusion.py` | Reciprocal Rank Fusion ($k=60$) & candidate deduplication | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/retrieval/hybrid.py` | Hybrid retriever coordinating Dense + BM25 + RRF | `02_CODEBASE_DEEP_DIVE.md`, `03_ARCHITECTURE_AND_FLOWS.md` |
| `src/retrieval/router.py` | Rule-based Query Intent Router & comparative capping | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/retrieval/reranker.py` | Optional Cross-Encoder reranker with RAM safeguards | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/retrieval/models.py` | Pydantic schemas: `RetrievedChunk`, `HybridRetrievalConfig` | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/retrieval/cli.py` | Command-line interface for testing query retrieval | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

#### Generation Subsystem (`src/generation/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/generation/__init__.py` | Module exports for answer generation and providers | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/generation/generator.py` | Answer generator, JSON parser, and citation validator | `02_CODEBASE_DEEP_DIVE.md`, `07_SECURITY_AND_CONFIGURATION.md` |
| `src/generation/prompts.py` | Defensive prompt templates with XML boundaries | `02_CODEBASE_DEEP_DIVE.md`, `07_SECURITY_AND_CONFIGURATION.md` |
| `src/generation/models.py` | Pydantic schemas: `Citation`, `GeneratedAnswer`, `AnswerStatus` | `02_CODEBASE_DEEP_DIVE.md`, `05_DATABASE_AND_DATA.md` |
| `src/generation/cli.py` | Command-line interface for testing question answering | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `src/generation/providers/base.py` | Abstract Base Class `LLMProvider` definition | `02_CODEBASE_DEEP_DIVE.md` |
| `src/generation/providers/factory.py` | `LLMProviderFactory` implementing auto-fallback | `02_CODEBASE_DEEP_DIVE.md`, `07_SECURITY_AND_CONFIGURATION.md` |
| `src/generation/providers/mock.py` | Deterministic synthetic mock provider for testing | `02_CODEBASE_DEEP_DIVE.md` |
| `src/generation/providers/gemini_provider.py` | Google Gemini Cloud API provider implementation | `02_CODEBASE_DEEP_DIVE.md`, `07_SECURITY_AND_CONFIGURATION.md` |
| `src/generation/providers/groq_provider.py` | Groq Cloud Llama-3 API provider implementation | `02_CODEBASE_DEEP_DIVE.md`, `07_SECURITY_AND_CONFIGURATION.md` |
| `src/generation/providers/openai_provider.py` | OpenAI GPT-4o API provider implementation | `02_CODEBASE_DEEP_DIVE.md`, `07_SECURITY_AND_CONFIGURATION.md` |

#### Evaluation Framework (`src/evaluation/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/evaluation/__init__.py` | Module exports for evaluation components | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/evaluation/evaluator.py` | Retrieval metrics engine (Hit@k, MRR@k, NDCG@k) | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/evaluation/generation_evaluator.py` | Answer quality evaluator (citations, abstentions, latency) | `02_CODEBASE_DEEP_DIVE.md`, `06_ALGORITHMS_AND_ML.md` |
| `src/evaluation/ground_truth.py` | Ground truth dataset loader and schema validator | `02_CODEBASE_DEEP_DIVE.md` |
| `src/evaluation/reporting.py` | Automated Markdown & JSON benchmark report generator | `02_CODEBASE_DEEP_DIVE.md` |
| `src/evaluation/groq_generation_runner.py` | Automated Groq evaluation benchmark runner | `02_CODEBASE_DEEP_DIVE.md` |
| `src/evaluation/ollama_generation_runner.py` | Automated Ollama evaluation benchmark runner | `02_CODEBASE_DEEP_DIVE.md` |
| `src/evaluation/models.py` | Pydantic evaluation schemas and metrics data models | `02_CODEBASE_DEEP_DIVE.md` |
| `src/evaluation/cli.py` | Command-line interface for running evaluation benchmarks | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

#### API & Web Backend (`src/api/` & `src/ui/`)
| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `src/api/__init__.py` | FastAPI application package initializer | `01_FOLDER_AND_FILE_GUIDE.md` |
| `src/api/app.py` | FastAPI application setup, CORS middleware, routing | `02_CODEBASE_DEEP_DIVE.md`, `04_FRONTEND_AND_API.md` |
| `src/api/dependencies.py` | Dependency injection: singleton retriever and generator | `02_CODEBASE_DEEP_DIVE.md`, `04_FRONTEND_AND_API.md` |
| `src/api/schemas.py` | REST API request/response Pydantic models | `04_FRONTEND_AND_API.md` |
| `src/api/routes/research.py` | `/api/research/ask` question answering endpoint | `04_FRONTEND_AND_API.md` |
| `src/api/routes/papers.py` | `/api/papers` paper catalog & details endpoints | `04_FRONTEND_AND_API.md` |
| `src/api/routes/system.py` | `/api/system/health` and `/api/system/telemetry` endpoints | `04_FRONTEND_AND_API.md`, `07_SECURITY_AND_CONFIGURATION.md` |
| `src/ui/app.py` | Streamlit interactive UI entrypoint | `02_CODEBASE_DEEP_DIVE.md`, `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `src/ui/service.py` | Streamlit backend service bridge | `02_CODEBASE_DEEP_DIVE.md` |

---

### 1.3 Next.js 15 Frontend (`frontend/`)

| File Path | Primary Responsibility | Covered In Documentation |
| :--- | :--- | :--- |
| `frontend/src/app/page.tsx` | Main App Router page managing top-level tabs & state | `04_FRONTEND_AND_API.md` |
| `frontend/src/app/layout.tsx` | Global HTML shell, fonts, and dark theme provider | `04_FRONTEND_AND_API.md` |
| `frontend/src/app/globals.css` | Tailwind CSS v4 styling rules and CSS variables | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/layout/Header.tsx` | App header with live provider pills and telemetry trigger | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/layout/Sidebar.tsx` | Left-hand navigation sidebar (Workspace, Library, System, About) | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/layout/SettingsModal.tsx` | Configuration dialog for provider keys and model selection | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/research/QueryComposer.tsx` | Research question input with query mode toggles | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/research/AnswerCard.tsx` | Synthesized answer renderer, status badges, and citations | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/research/CitationCard.tsx` | Individual verified citation card with verbatim excerpt | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/research/ContextInspector.tsx` | Collapsible drawer inspecting all retrieved chunks & scores | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/research/EmptyWorkspace.tsx` | Initial empty-state screen with quick starter questions | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/research/LoadingSkeleton.tsx` | Animated pulse skeleton state during generation | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/papers/PaperTable.tsx` | Academic paper library table with search & filtering | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/papers/CorpusList.tsx` | Chunk-level breakdown and page index list | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/papers/IngestModal.tsx` | Modal dialog for paper ingestion parameters | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/system/SystemOverview.tsx` | System health dashboard, vector count, and latency charts | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/system/ProviderGrid.tsx` | Grid displaying live status of Gemini, Groq, OpenAI, Ollama | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/system/ArchitectureCard.tsx` | Visual diagram of the RIE pipeline in the telemetry tab | `04_FRONTEND_AND_API.md` |
| `frontend/src/components/about/ProjectOverview.tsx` | Technical architecture overview, metrics, and paper list | `04_FRONTEND_AND_API.md` |
| `frontend/src/lib/api.ts` | Type-safe HTTP client communicating with FastAPI | `04_FRONTEND_AND_API.md` |
| `frontend/src/lib/types.ts` | TypeScript interfaces mirroring backend Pydantic models | `04_FRONTEND_AND_API.md` |
| `frontend/src/lib/utils.ts` | UI utility functions (Tailwind class merging, formatting) | `04_FRONTEND_AND_API.md` |
| `frontend/package.json` | Frontend dependencies, scripts, and package versions | `07_SECURITY_AND_CONFIGURATION.md` |
| `frontend/next.config.ts` | Next.js runtime configuration | `01_FOLDER_AND_FILE_GUIDE.md` |

---

### 1.4 Test Suite (`tests/`)

| Test File | Verification Scope | Covered In Documentation |
| :--- | :--- | :--- |
| `tests/conftest.py` | Pytest fixtures: mock Qdrant client, sample chunks | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_ingestion.py` | PDF parsing, page extraction, and metadata decoupling | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_chunking.py` | Boundary-preserving sentence chunking and overlap logic | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_indexing.py` | MiniLM embeddings, Qdrant upserts, and UUIDv5 stability | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_retrieval.py` | Dense vector search accuracy and score thresholds | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_hybrid_retrieval.py` | Reciprocal Rank Fusion ($k=60$) and BM25 integration | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_diversified_retrieval.py`| Per-document chunk capping and candidate backfill | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_query_intent_routing.py` | Regex intent classification (Comparative vs. Standard) | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_reranker.py` | Optional Cross-Encoder reranking and RAM safeguards | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_generation.py` | LLM synthesis, JSON parsing, and citation verification | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_answer_quality_evaluation.py` | 6 benchmark dimensions across 28 test questions | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_api_endpoints.py` | FastAPI endpoint contracts (`/ask`, `/papers`, `/health`) | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_api_live_integration.py` | Live HTTP client integration test against FastAPI | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_groq_generation_runner.py` | Groq provider integration and error recovery | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_groq_json_schema_handling.py` | Groq JSON output parsing and repair heuristics | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_groq_rate_limit_handling.py` | HTTP 429 backoff and retry behavior on Groq Cloud | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_ollama_generation_runner.py` | Local Ollama provider HTTP integration tests | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_demo_workflow.py` | End-to-end interactive demo workflow verification | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `tests/test_ui_app.py` | Streamlit UI component rendering and mock queries | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

---

### 1.5 Execution Scripts (`scripts/`)

| Script File | Execution Purpose | Covered In Documentation |
| :--- | :--- | :--- |
| `scripts/run_demo.py` | Interactive terminal-based RAG question answering demo | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_qa.py` | Single-query CLI runner with full provenance output | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_ingestion.py` | Batch PDF ingestion runner across `data/papers/` | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_chunking.py` | Batch sentence chunker runner across extracted pages | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_indexing.py` | Vector store indexer generating embeddings for Qdrant | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_retrieval.py` | Standalone CLI query retrieval diagnostic script | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_evaluation.py` | Retrieval metrics evaluation across benchmark queries | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_generation_evaluation.py` | Full answer quality, citation validity & latency benchmark | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_answer_quality_evaluation.py` | Focused evaluation runner for citation and abstention checks | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |
| `scripts/run_dense_hybrid_retrieval_comparison.py` | Direct side-by-side comparison of Dense vs. Hybrid RRF | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_dense_retrieval_diagnostic.py` | Diagnostic script identifying dense retrieval failure modes | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_diversified_retrieval_evaluation.py` | Benchmark evaluating multi-document representation | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_query_intent_routing_evaluation.py` | Accuracy benchmark for the rule-based Intent Router | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_cross_encoder_reranking_evaluation.py` | Experimental CPU Cross-Encoder reranking benchmark | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_cross_paper_evaluation.py` | Cross-paper comparative query evaluation runner | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_groq_generation_evaluation.py` | Automated evaluation runner executing against Groq Cloud | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_live_gemini_generation_evaluation.py` | Automated evaluation runner executing against Gemini API | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_ollama_generation_evaluation.py` | Automated evaluation runner executing against local Ollama | `06_ALGORITHMS_AND_ML.md` |
| `scripts/run_smoke_test.py` | Rapid smoke test verifying live generation across providers | `08_SETUP_TESTING_AND_DEPLOYMENT.md` |

---

## 2. Explicit Operational Exclusions

The following directories, temporary files, and binary artifacts are intentionally excluded from manual documentation inspection and version control:

1. **Version Control Internals (`.git/`):** Contains Git commit history, blobs, tree objects, and refs. Fully tracked via Git CLI and remote GitHub repository.
2. **Node.js Package Distribution (`frontend/node_modules/`):** Third-party compiled JavaScript packages. Fully specified and managed via `frontend/package.json` and `package-lock.json`.
3. **Next.js Build Output (`frontend/.next/`):** Ephemeral compilation cache, server bundles, and static page chunks produced by `npm run build`.
4. **Python Virtual Environments (`.venv/`, `venv/`, `env/`):** Local Python interpreter binaries and installed site-packages. Fully reproducible via `requirements.txt`.
5. **Python Compilation Caches (`__pycache__/`, `*.pyc`):** CPython bytecode caches generated at runtime.
6. **Pytest Caches (`.pytest_cache/`):** Test runner caching and step tracking metadata.
7. **Vector Database Lockfile (`data/qdrant/.lock`):** Temporary OS file lock acquired exclusively by the running RocksDB storage engine.

---

## 3. Final Verification Checklist

- [x] **Chapter 00:** High-level project purpose, problem statement, personas, and implementation status documented.
- [x] **Chapter 01:** Complete folder structure and file responsibility catalog written.
- [x] **Chapter 02:** In-depth module, class, function, and error-handling breakdown written.
- [x] **Chapter 03:** Architecture diagrams, Mermaid sequence flows, and 3 numbered end-to-end execution flows written.
- [x] **Chapter 04:** Next.js 15 frontend architecture, component tree, user journeys, and REST API specification table written.
- [x] **Chapter 05:** Qdrant RocksDB vector store, BM25 JSON corpus, schemas, ER diagram, and passage lifecycle written.
- [x] **Chapter 06:** Sentence chunking, embeddings, BM25, RRF fusion ($k=60$), intent router, and evaluation formulas documented.
- [x] **Chapter 07:** Threat model, prompt injection defenses, citation verification, environment variables, and dependencies cataloged.
- [x] **Chapter 08:** Prerequisites, local setup, service commands, testing procedures, production deployment, and troubleshooting guide written.
- [x] **Chapter 09:** Code quality audit, technical debt, RocksDB lock contention, and prioritized engineering roadmap documented.
- [x] **Chapter 10:** 2-minute architectural pitch, technical Q&A, trade-off matrix, and 45-minute whiteboarding guide written.
- [x] **Chapter 11:** Complete file-by-file coverage matrix, operational exclusions, and audit verification checklist written.
- [x] **Hub (README.md):** Directory navigation hub and role-based reading pathways established.

