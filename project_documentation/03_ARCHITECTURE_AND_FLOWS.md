# 03 — System Architecture and Execution Flows

This document details the architectural design and end-to-end execution flows of the **Research Intelligence Engine (RIE)**. Every diagram and numbered trace directly reflects the code paths implemented in `src/` and `frontend/`.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer ["1. Presentation Layer"]
        NextJS["Next.js 15 Web Application<br/>(React 19, TypeScript, Tailwind)"]
        Streamlit["Streamlit Dashboard<br/>(Standalone Fallback UI)"]
    end

    subgraph APILayer ["2. API & Service Façade"]
        FastAPI["FastAPI Backend Server<br/>(Uvicorn on Port 8000)"]
        Service["ResearchEngineService<br/>(Central Singleton Façade)"]
    end

    subgraph CoreRAG ["3. Retrieval & Synthesis Engine"]
        Router["QueryIntentClassifier<br/>(Standard vs. Comparative)"]
        Hybrid["HybridRetriever<br/>(Dense + Okapi BM25)"]
        RRF["Reciprocal Rank Fusion<br/>(RRF k=60)"]
        Generator["AnswerGenerator<br/>(Strict Schema & Provenance)"]
    end

    subgraph StorageLayer ["4. Dual-Storage & ML Models"]
        Qdrant[("Local Qdrant Store<br/>(RocksDB 384-d Vectors)")]
        BM25Data[("Okapi BM25 Index<br/>(corpus_chunks.json)")]
        SBERT["SentenceTransformer<br/>(all-MiniLM-L6-v2)"]
    end

    subgraph Providers ["5. LLM Provider Tier"]
        MockLLM["Mock Provider (Deterministic CI)"]
        Gemini["Google Gemini (gemini-2.5-flash)"]
        OpenAI["OpenAI (gpt-4o-mini / gpt-4o)"]
        Groq["Groq Cloud (llama-3.3-70b)"]
        Ollama["Local Ollama (gemma3:4b)"]
    end

    NextJS -->|HTTP REST JSON| FastAPI
    Streamlit -->|In-Process Service Call| Service
    FastAPI --> Service
    Service --> Router
    Router --> Hybrid
    Hybrid -->|Dense Search| SBERT
    SBERT --> Qdrant
    Hybrid -->|Sparse Keyword Search| BM25Data
    Hybrid --> RRF
    RRF --> Generator
    Generator --> Providers
    Providers --> Generator
    Generator --> Service
    Service --> FastAPI
    FastAPI --> NextJS
```

---

## 2. Execution Flow 1: End-to-End Research Question Answering

This is the primary user journey: a user submits a natural language question and receives an evidence-grounded answer with verified page citations.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Next.js (QueryComposer)
    participant API as FastAPI (/api/research/ask)
    participant Svc as ResearchEngineService
    participant Router as QueryIntentClassifier
    participant Ret as HybridRetriever
    participant Dense as DenseRetriever & Qdrant
    participant BM25 as BM25Retriever
    participant Gen as AnswerGenerator
    participant LLM as LLM Provider
    participant Card as Next.js (AnswerCard)

    User->>UI: Enters question & clicks "Ask Question"
    UI->>API: POST /api/research/ask (question, provider, mode, top_k)
    API->>Svc: ask_question(question, provider_name, retrieval_mode, top_k)
    Svc->>Router: classify_query_intent(question)
    Router-->>Svc: QueryIntentResult (standard or comparative)
    Svc->>Ret: retrieve(question, top_k=5)
    par Parallel Candidate Search
        Ret->>Dense: retrieve(question, top_k=20)
        Dense-->>Ret: Top 20 dense candidates (Cosine similarity)
    and
        Ret->>BM25: search(question, top_k=20)
        BM25-->>Ret: Top 20 lexical candidates (BM25 scores)
    end
    Ret->>Ret: Reciprocal Rank Fusion (RRF k=60)
    Note over Ret: If comparative intent: apply per-paper candidate cap
    Ret-->>Svc: Top-5 fused RetrievedChunk instances
    Svc->>Gen: generate(question, retrieval_result)
    Gen->>Gen: build_generation_messages() (isolated passage blocks)
    Gen->>LLM: generate(messages, response_format="json_object")
    LLM-->>Gen: Raw model JSON response
    Gen->>Gen: _validate_and_build_citations() (verify chunk IDs & reconcile excerpts)
    Gen-->>Svc: GeneratedAnswer (status, answer, citations, latency)
    Svc-->>API: AskQuestionResponse
    API-->>UI: HTTP 200 JSON payload
    UI->>Card: Renders answer, status badge, latency timer, citation pills
```

### Step-by-Step Breakdown:
1. **Input Submission:** In `QueryComposer.tsx`, the user enters a research question (e.g., *"What generator model does RAG use?"*).
2. **API Dispatch:** `api.askQuestion()` in `frontend/src/lib/api.ts` makes a `POST` request to `http://localhost:8000/api/research/ask`.
3. **Route Handling:** In `src/api/routes/research.py`, `ask_research_question()` validates that the query is non-empty and invokes the shared `ResearchEngineService`.
4. **Intent Classification:** In `src/retrieval/router.py`, `classify_query_intent()` inspects keywords to determine if multi-paper balancing is needed.
5. **Dual Retrieval:** `HybridRetriever.retrieve()` in `src/retrieval/hybrid.py` executes:
   * `DenseRetriever`: Encodes query into 384-d vector via `all-MiniLM-L6-v2` and runs cosine search in Qdrant (`data/qdrant/`).
   * `BM25Retriever`: Tokenizes query and computes Okapi BM25 scores across `data/chunked/corpus_chunks.json`.
6. **Reciprocal Rank Fusion (RRF):** `reciprocal_rank_fusion()` in `src/retrieval/fusion.py` computes combined scores using $k=60$ and ranks candidates.
7. **Prompt Construction:** In `src/generation/prompts.py`, `format_evidence_block()` formats top-$k$ passages into unambiguous delimiters:
   `=== PASSAGE [P1_p03_c02] (Paper: P1, Page: 3) ===`
8. **Model Generation:** Provider adapter (`gemini`, `openai`, `groq`, `ollama`, or `mock`) calls the LLM with `response_format="json_object"`.
9. **Citation Validation & Excerpt Reconciliation:** `AnswerGenerator._validate_and_build_citations()`:
   * Validates each emitted `chunk_id` against retrieved passages; **rejects hallucinated chunk IDs**.
   * Checks model excerpt against source text; if modified, extracts verbatim passage prefix.
   * Populates physical page numbers directly from chunk metadata.
10. **UI Rendering:** In `frontend/src/app/page.tsx`, `AnswerCard` displays the response, latency, and interactive citation badges; `ContextInspector` allows auditing all retrieved context passages.

---

## 3. Execution Flow 2: Deterministic Abstention Safeguard Flow

RIE enforces deterministic abstention to ensure zero hallucination when evidence is absent or out of scope.

```mermaid
flowchart TD
    Start["User Submits Query"] --> Retrieval["Execute Hybrid Retrieval"]
    Retrieval --> CheckChunks{"Retrieved Chunks<br/>Count > 0?"}

    CheckChunks -- No --> ImmediateAbstain["IMMEDIATE ABSTENTION (Zero-Cost)<br/>status: 'abstained_insufficient_evidence'<br/>citations: []<br/>latency: < 5ms"]
    ImmediateAbstain --> ReturnResponse["Return GeneratedAnswer to User"]

    CheckChunks -- Yes --> CallLLM["Invoke LLM Provider with Context"]
    CallLLM --> CheckStatus{"Model JSON<br/>Status?"}

    CheckStatus -- "abstained_insufficient_evidence" --> EnforceZeroCites1["Enforce citations = []<br/>Confidence Note: 'Explicit model abstention'"]
    CheckStatus -- "abstained_out_of_scope" --> EnforceZeroCites2["Enforce citations = []<br/>Confidence Note: 'Query out of scientific scope'"]
    CheckStatus -- "answered" --> ValidateCites["Validate Chunk IDs in Retrieved Context"]

    ValidateCites --> AnyValid{"Valid Chunks<br/>Cited?"}
    AnyValid -- No --> FlagUnsupported["Status: ERROR / Unsubstantiated<br/>citations: []"]
    AnyValid -- Yes --> ReconcileExcerpts["Reconcile Verbatim Excerpts<br/>Populate Physical Page Numbers"]

    EnforceZeroCites1 --> ReturnResponse
    EnforceZeroCites2 --> ReturnResponse
    FlagUnsupported --> ReturnResponse
    ReconcileExcerpts --> ReturnResponse
```

### Key Safety Invariants Enforced in Code:
* **Immediate Zero-Cost Abstention:** In `src/generation/generator.py` line 395: If `retrieval_result.chunks` is empty or no chunk passes the similarity threshold, the engine immediately returns `AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE` with empty citations without invoking the LLM provider.
* **Zero Phantom Citations on Abstention:** In `src/generation/generator.py` and unit tests: Whenever `status` is an abstention variant, the citation list is strictly emptied.
* **Hallucination Rejection:** In `src/generation/generator.py` line 145: Any citation referencing a `chunk_id` not present in `chunks_map` is discarded and flagged in `confidence_notes`.

---

## 4. Execution Flow 3: Document Ingestion & Dynamic Index Synchronization

When a new PDF is added via the Web UI (`IngestModal.tsx`) or API (`POST /api/papers/upload`), RIE executes an end-to-end extraction and index synchronization pipeline.

```mermaid
sequenceDiagram
    autonumber
    actor Admin as User / Admin
    participant UI as Next.js (IngestModal)
    participant Route as FastAPI (/api/papers/upload)
    participant Svc as ResearchEngineService
    participant Loader as PDFLoader (PyMuPDF)
    participant Chunker as TextChunker
    participant Indexer as IndexingPipeline
    participant Qdrant as QdrantStoreManager
    participant FileSys as corpus_chunks.json

    Admin->>UI: Selects PDF & clicks "Ingest Document"
    UI->>Route: POST /api/papers/upload (Multipart FormData)
    Route->>Route: Saves upload to secure temp directory
    Route->>Svc: ingest_paper(pdf_path, doc_id, doc_title, dry_run=False)
    Svc->>Loader: load_document(pdf_path, metadata)
    Loader->>Loader: Page-by-page extraction & physical page indexing
    Loader-->>Svc: DocumentIngestionResult (pages list)
    Svc->>Chunker: chunk_document_pages(pages)
    Chunker->>Chunker: Sentence boundary segmentation & token budgeting
    Chunker-->>Svc: List[DocumentChunk] (deterministic IDs)
    alt Dry Run Mode
        Svc-->>Route: IngestionReport (previews chunks without saving)
    else Live Ingestion
        Svc->>Indexer: index_chunks(chunks)
        Indexer->>Indexer: Compute 384-d dense embeddings (all-MiniLM-L6-v2)
        Indexer->>Qdrant: upsert_chunks(collection_name, chunks, vectors)
        Qdrant-->>Indexer: Points committed to RocksDB
        Svc->>FileSys: _sync_chunks_to_file() (creates .bak & writes corpus_chunks.json)
        Svc-->>Route: IngestionReport (pages, chunks_indexed, total_points)
    end
    Route-->>UI: IngestionResponse JSON
    UI->>UI: Refreshes PaperTable and SystemOverview
```

### Safety Features of Ingestion:
1. **Physical Page Number Preservation:** Physical page indices ($1, 2, 3, \dots$) are attached during the initial PyMuPDF pass (`loader.py`), guaranteeing that chunk metadata carries verifiable page coordinates into Qdrant.
2. **Safe File Synchronization with Backup:** In `src/ui/service.py` line 312: Before writing newly ingested chunks to `data/chunked/corpus_chunks.json`, the service writes an automatic backup to `data/chunked/corpus_chunks.json.bak` via `shutil.copy2()`, preventing data loss on disk full or write interruptions.
3. **Idempotent Point IDs:** In `src/indexing/vector_store.py`: Point IDs in Qdrant are generated deterministically using SHA-256 hashes of the chunk ID, ensuring that re-ingesting a document updates existing points rather than duplicating them.

