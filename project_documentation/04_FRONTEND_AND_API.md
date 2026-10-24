# 04 — Frontend Interface and REST API Analysis

This document provides a technical analysis of the **Next.js 15 web application** and the **FastAPI REST API contracts**.

---

## 1. Next.js 15 Frontend Architecture

### 1.1 Architectural Overview
The frontend is built using **Next.js 15** (React 19, App Router) and written in **TypeScript**. It provides a responsive, single-page application (SPA) layout with dark-mode styling implemented via **Tailwind CSS** and icons from **Lucide React**.

* **Root Layout:** `frontend/src/app/layout.tsx` wraps the app with Geist font definitions and global dark-mode styles.
* **Main Dashboard:** `frontend/src/app/page.tsx` (437 lines) manages application view state, multi-turn research sessions, settings persistence, and API data fetching.
* **Component Directory:** `frontend/src/components/` is organized into modular functional directories: `research/`, `papers/`, `system/`, `about/`, and `layout/`.

### 1.2 State Management & Persistence
The frontend operates without external state management libraries (such as Redux or Zustand), using React `useState`, `useCallback`, and browser `localStorage`:
* **Settings Persistence (`rie_settings`):** Stores user preferences in `localStorage`: active provider (`mock`, `gemini`, `openai`, `groq`, `ollama`), model override string, retrieval mode (`hybrid` or `dense`), `top_k` (default 5), and `enable_intent_routing` boolean.
* **Session History Persistence (`rie_sessions`):** Multi-turn research sessions containing past query-answer pairs are persisted in `localStorage`. Users can create new sessions, switch between sessions, or delete individual sessions from the sidebar.
* **Clean Mount Invariant:** On initial page load, `activeSessionId` is set to `null` so the user is greeted by the curated `EmptyWorkspace` benchmark cards. Past queries are only displayed when the user explicitly clicks a past session in the sidebar.

---

## 2. Component Hierarchy & Walkthrough

```text
frontend/src/app/page.tsx (Main Shell)
├── Sidebar.tsx (Navigation & Session History)
├── Header.tsx (Top Bar & Settings Trigger)
├── SettingsModal.tsx (Engine Configuration Dialog)
│
├── [Tab: 'research'] -> Research Workspace
│   ├── QueryComposer.tsx (Input Bar, Provider Quick-Select, Submit Button)
│   ├── EmptyWorkspace.tsx (Benchmark Query Cards: Factual, Comparative, Out-of-Scope)
│   ├── LoadingSkeleton.tsx (Animated Shimmer during Generation)
│   └── AnswerCard.tsx (Grounded Answer Display)
│       ├── CitationCard.tsx (Page Badges & Verbatim Source Excerpts)
│       └── ContextInspector.tsx (Slide-Out Drawer Auditing All Top-k Chunks)
│
├── [Tab: 'papers'] -> Literature & Ingestion
│   ├── PaperTable.tsx (Indexed Documents, Page Counts, Chunks)
│   ├── CorpusList.tsx (Benchmark PDFs with One-Click Ingestion)
│   └── IngestModal.tsx (File Upload Dialog with Dry-Run Toggle)
│
├── [Tab: 'status'] -> System Telemetry
│   ├── SystemOverview.tsx (Vector Points, Collection Stats)
│   ├── ProviderGrid.tsx (Live Health & Models for Gemini, OpenAI, Groq, Ollama)
│   └── ArchitectureCard.tsx (4-Stage Pipeline Technical Diagram)
│
└── [Tab: 'about'] -> Project Overview
    └── ProjectOverview.tsx (Technical Deep Dive, Measured Metrics, Architecture)
```

### Component Details:
* **`QueryComposer.tsx`:** Provides an accessible form with an auto-expanding input textarea, keyboard shortcut (`Enter` to submit, `Shift+Enter` for newline), quick provider dropdown, and submit button that disables during loading.
* **`AnswerCard.tsx`:** Renders synthesized responses with a status pill reflecting the `AnswerStatus` enum (`answered` in green, `abstained` in amber, `error` in red). Displays round-trip latency in milliseconds, model name, and interactive citation badges.
* **`CitationCard.tsx`:** Formats each verified citation as an academic reference: Paper ID (e.g., `P1`), Title, Physical Page Number, Chunk ID, and the verified verbatim excerpt.
* **`ContextInspector.tsx`:** A slide-out drawer allowing researchers to audit every passage retrieved by the hybrid engine, inspecting similarity scores and chunk boundaries to verify why an answer was generated or why the system abstained.
* **`PaperTable.tsx` & `CorpusList.tsx`:** Display collection inventories and allow researchers to re-index seminal PDFs with a single click.
* **`IngestModal.tsx`:** Features drag-and-drop PDF upload with client-side mime-type checking and a "Dry Run" toggle to validate chunk boundaries without modifying the persistent database.

---

## 3. Major User Journeys

### User Journey 1: Executing Curated Benchmark Queries
1. User opens the application at `http://localhost:3000`. The workspace renders `EmptyWorkspace` with 4 prompt cards.
2. User clicks a benchmark card (e.g., *"What generator model does RAG use?"*).
3. The prompt is populated into `QueryComposer` and automatically submitted.
4. `LoadingSkeleton` renders an animated shimmer.
5. `AnswerCard` appears showing the answer (*"RAG uses BART-large (400M parameters)..."*), green `Answered` badge, latency (e.g., `18.5ms`), and citation `[P1, Page 3, Chunk P1_p03_c02]`.
6. User clicks *"Inspect Retrieved Context (5 Chunks)"* to review the exact ranking and scores of all candidate passages in `ContextInspector`.

### User Journey 2: Out-of-Scope Query & Deterministic Abstention
1. User enters an unanswerable query (e.g., *"What is the computational cost in GPU-hours of training a Self-RAG model from scratch?"*).
2. The hybrid retrieval engine retrieves candidates, but the generation engine detects that the required facts are absent.
3. The UI renders an amber `Abstained (Insufficient Evidence)` badge, explicitly informing the user that the corpus does not substantiate the answer.
4. **Zero Citations Guarantee:** The citation list is completely empty, ensuring no phantom sources are cited.

### User Journey 3: Custom PDF Ingestion & Real-Time Re-Indexing
1. User navigates to the **Paper Library** tab (`papers`).
2. User clicks *"Upload PDF Document"*, triggering `IngestModal`.
3. User drags and drops a new research paper PDF (e.g., `P6_ColBERT.pdf`), specifies Document ID `P6`, and clicks *"Ingest Document"*.
4. FastAPI extracts pages, chunks text, generates embeddings, upserts vectors to Qdrant, and updates `corpus_chunks.json`.
5. The Paper Library table updates dynamically showing `P6` with its extracted page count and chunk total.

---

## 4. RESTful API Reference Specification

The FastAPI backend server (`server.py`, `src/api/app.py`) provides the following endpoints:

| Method | Endpoint Path | Summary & Purpose | Auth | Request Body / Parameters | Response Schema | Source File |
|---|---|---|:---:|---|---|---|
| `GET` | `/api/system/health` | Lightweight service health probe. | None | None | `HealthResponse` | `src/api/routes/system.py` |
| `GET` | `/health` | Root health probe alias for container health checks. | None | None | `HealthResponse` | `src/api/app.py` |
| `GET` | `/api/system/status` | Real-time vector store statistics and provider configuration status. | None | None | `SystemStatusResponse` | `src/api/routes/system.py` |
| `GET` | `/api/papers` | Retrieves all documents currently indexed in the active research corpus. | None | None | `List[PaperSummaryResponse]` | `src/api/routes/papers.py` |
| `GET` | `/api/papers/corpus` | Lists benchmark PDFs available in `evaluation/corpus/papers/`. | None | None | `List[CorpusPDFResponse]` | `src/api/routes/papers.py` |
| `POST` | `/api/papers/ingest` | Triggers ingestion or dry-run validation of an existing corpus PDF. | None | `IngestPaperRequest` (JSON) | `IngestionResponse` | `src/api/routes/papers.py` |
| `POST` | `/api/papers/upload` | Uploads and ingests a custom PDF file via multipart form. | None | `file: UploadFile`, `dry_run: bool`, `doc_id`, `doc_title` | `IngestionResponse` | `src/api/routes/papers.py` |
| `POST` | `/api/research/ask` | Executes end-to-end evidence retrieval, synthesis, and citation validation. | None | `AskQuestionRequest` (JSON) | `AskQuestionResponse` | `src/api/routes/research.py` |

---

## 5. Detailed API Endpoint Contracts

### 5.1 `POST /api/research/ask`
* **Purpose:** Primary research Q&A endpoint.
* **Request Payload (`AskQuestionRequest`):**
  ```json
  {
    "question": "What generator model does RAG use?",
    "provider": "mock",
    "model": null,
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "min_score": null,
    "enable_intent_routing": false
  }
  ```
* **Validation Rules:**
  * `question`: Required string, stripped. HTTP 400 returned if empty or whitespace-only.
  * `retrieval_mode`: Must be `'hybrid'` or `'dense'`.
  * `top_k`: Integer between 1 and 50 (default: 5).
* **Response Payload (`AskQuestionResponse`):**
  ```json
  {
    "question": "What generator model does RAG use?",
    "answer": "RAG models use BART-large with 400M parameters as the generator component.",
    "status": "answered",
    "citations": [
      {
        "chunk_id": "P1_p03_c02",
        "document_id": "P1",
        "document_title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "physical_page_number": 3,
        "printed_page_number": 3,
        "excerpt": "uses BART-large with 400M parameters"
      }
    ],
    "retrieved_chunks": [
      {
        "chunk_id": "P1_p03_c02",
        "document_id": "P1",
        "document_title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "physical_page_number": 3,
        "printed_page_number": 3,
        "text": "The generator component is BART-large...",
        "score": 0.0328
      }
    ],
    "model_name": "mock-model",
    "provider_name": "mock",
    "retrieval_mode": "hybrid",
    "latency_ms": 18.5,
    "confidence_notes": null,
    "prompt_tokens": 850,
    "completion_tokens": 65,
    "total_tokens": 915
  }
  ```
* **Error Status Codes:**
  * `400 Bad Request`: Empty or invalid question text.
  * `500 Internal Server Error`: Unhandled backend exception during generation.

---

### 5.2 `GET /api/system/status`
* **Purpose:** Telemetry probe providing collection point counts and auditing LLM provider keys.
* **Response Payload (`SystemStatusResponse`):**
  ```json
  {
    "collection_name": "research_corpus",
    "total_points": 396,
    "indexed_papers_count": 5,
    "papers": [
      {
        "document_id": "P1",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "total_pages": 12,
        "total_chunks": 88
      }
    ],
    "providers": {
      "mock": { "configured": true, "default_model": "mock-model" },
      "gemini": { "configured": true, "default_model": "gemini-2.5-flash" },
      "groq": { "configured": true, "default_model": "llama-3.3-70b-versatile" },
      "openai": { "configured": false, "default_model": "gpt-4o-mini" },
      "ollama": { "configured": true, "default_model": "gemma3:4b" }
    }
  }
  ```
* **Security Feature:** Provider API keys are never exposed in responses; only boolean `configured` flags and default model strings are returned.

---

### 5.3 `POST /api/papers/upload`
* **Purpose:** Uploads a custom PDF document via multipart form, extracting pages, chunking text, generating embeddings, and upserting points into Qdrant.
* **Request Format:** Multipart Form Data (`file: UploadFile`, `dry_run: Form(False)`, `doc_id: Form(None)`, `doc_title: Form(None)`).
* **Validation:** Rejects non-PDF files (`content_type != application/pdf` or extension `!= .pdf`) with HTTP 400.
* **Response Payload (`IngestionResponse`):**
  ```json
  {
    "success": true,
    "document_id": "P6",
    "title": "ColBERT: Efficient and Effective Passage Search",
    "filename": "P6_ColBERT.pdf",
    "pages_extracted": 10,
    "chunks_created": 45,
    "chunks_indexed": 45,
    "total_collection_points": 441,
    "error_message": null,
    "sample_chunks": [
      {
        "chunk_id": "P6_p01_c01",
        "page": 1,
        "words": 142,
        "preview": "In this paper we propose ColBERT, a novel ranking model..."
      }
    ]
  }
  ```

