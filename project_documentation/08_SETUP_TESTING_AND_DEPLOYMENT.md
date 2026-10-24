# 08 — Setup, Testing & Deployment Guide

## Executive Summary

The **Research Intelligence Engine (RIE)** is engineered to operate seamlessly across development, evaluation, and production environments. This guide provides comprehensive, step-by-step procedures for local environment provisioning, service startup (FastAPI, Next.js, and Streamlit), CLI operations, unit and integration test execution, production deployment patterns, and operational troubleshooting.

---

## 1. System Prerequisites & Environment Matrix

| Requirement | Minimum Version | Recommended | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | `3.10.0` | `3.11.x` | Required for PyTorch, SentenceTransformers, and FastAPI. |
| **Node.js** | `18.17.0` | `20.x LTS` | Required for Next.js 15 App Router and React 19. |
| **npm** | `9.0.0` | `10.x` | Package manager for frontend dependencies. |
| **Operating System** | Windows 10/11, Ubuntu 22.04 LTS, or macOS 13+ | Windows 11 / Linux | Windows PowerShell and Linux bash fully supported. |
| **System Memory (RAM)** | 8 GB | 16 GB+ | 4 GB budget for MiniLM and PyTorch CPU tensors. |
| **Local LLM (Optional)**| Ollama `0.1.30+` | Ollama with `gemma3:4b` | Required only if running local inference offline without cloud keys. |

---

## 2. Local Development Setup

### Step 2.1: Clone the Repository
```bash
git clone https://github.com/Sanchitk99/research-intelligence-engine.git
cd research-intelligence-engine
```

### Step 2.2: Python Virtual Environment Setup

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip and install backend dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**On Linux / macOS (Bash):**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install backend dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2.3: Environment Configuration (`.env`)
Create a `.env` file in the project root by copying the template or specifying your preferred provider keys:

```ini
# Research Intelligence Engine - Environment Configuration

# Provider Selection: auto, gemini, groq, openai, ollama, or mock
LLM_PROVIDER=auto

# Upstream Cloud API Keys (Provide at least one for live cloud inference)
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Local Ollama Inference (If running local offline models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# Server & Network Configuration
PORT=8000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> [!NOTE]
> If no API keys or Ollama instance are configured, RIE defaults automatically to `MockProvider`, allowing full end-to-end testing, UI exploration, and retrieval verification without external dependencies or cloud charges.

### Step 2.4: Frontend Installation
Open a secondary terminal window and initialize the Next.js application:

```bash
cd frontend
npm install
```

---

## 3. Starting the Services

RIE provides two frontend interfaces: a modern **Next.js 15 Web Application** and a rapid-prototyping **Streamlit Dashboard**.

```
                   RIE RUNTIME ORCHESTRATION

    Terminal 1 (Backend API)          Terminal 2 (Primary UI)
   ┌────────────────────────┐       ┌────────────────────────┐
   │ FastAPI / Uvicorn      │       │ Next.js 15 App Router  │
   │ Port: 8000             │◄─────►│ Port: 3000             │
   │ Swagger: /docs         │       │ User Browser           │
   └────────────────────────┘       └────────────────────────┘

                   Terminal 3 (Alternative UI - Optional)
   ┌────────────────────────┐
   │ Streamlit Dashboard    │
   │ Port: 8501             │
   └────────────────────────┘
```

### Option A: Standard Full-Stack Architecture (FastAPI + Next.js)

**Terminal 1 — Backend API:**
```bash
# Activate venv if not active
.venv\Scripts\activate   # or source .venv/bin/activate

# Start FastAPI server on port 8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/system/health`

**Terminal 2 — Next.js 15 Frontend:**
```bash
cd frontend
npm run dev
```
- Web Application: `http://localhost:3000`

### Option B: Standalone Streamlit Prototyping Dashboard
If developing or testing retrieval algorithms without running Node.js:
```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```
- Streamlit Interface: `http://localhost:8501`

---

## 4. Pipeline Execution & CLI Maintenance Commands

All pipeline components can be invoked directly from the command line for corpus maintenance and batch processing.

### 4.1 Ingestion Pipeline CLI
Extracts text and layout metadata from raw PDF papers into `extracted_pages.json`:
```bash
python -m src.ingestion.cli \
  --papers-dir data/papers \
  --output data/ingested/extracted_pages.json
```

### 4.2 Sentence-Boundary Chunking CLI
Segments extracted pages into overlapping chunks adhering to sentence and paragraph boundaries:
```bash
python -m src.chunking.cli \
  --input data/ingested/extracted_pages.json \
  --output data/chunked/corpus_chunks.json \
  --chunk-size 1000 \
  --chunk-overlap 150
```

### 4.3 Qdrant Vector Indexing CLI
Computes 384-dimensional dense embeddings and indexes chunks into the embedded Qdrant vector database:
```bash
python -m src.indexing.cli \
  --input data/chunked/corpus_chunks.json \
  --collection research_corpus \
  --batch-size 32
```

### 4.4 Dual-Index Synchronization Script
Ensures that the embedded Qdrant vector store and the in-memory BM25 JSON corpus are perfectly synchronized:
```bash
python scripts/sync_corpus_indexes.py
```

### 4.5 Interactive Terminal Demo
Runs a terminal-based RAG question-answering session directly in PowerShell/bash:
```bash
python scripts/run_demo.py
```

---

## 5. Testing & Quality Assurance Guide

RIE features a comprehensive automated test suite covering unit tests, contract tests, evaluation benchmarks, and integration flows.

### 5.1 Running the Pytest Suite
Execute all unit and integration tests across the repository:
```bash
# Run entire test suite
pytest -v

# Run with stdout printing enabled
pytest -v -s
```

### 5.2 Focused Module Test Execution

```bash
# 1. Test Ingestion and PDF parsing
pytest tests/test_ingestion.py -v

# 2. Test Sentence-Boundary Chunking
pytest tests/test_chunking.py -v

# 3. Test Qdrant Vector Indexing
pytest tests/test_indexing.py -v

# 4. Test Dense, BM25, and Hybrid RRF Retrieval
pytest tests/test_retrieval.py -v

# 5. Test Answer Generation and Citation Validation
pytest tests/test_generation.py -v

# 6. Test FastAPI HTTP Endpoints
pytest tests/test_api.py -v

# 7. Test Answer Quality Evaluation Benchmark Suite
pytest tests/test_answer_quality_evaluation.py -v
```

### 5.3 Executing Evaluation Benchmarks

#### Retrieval Benchmark (Dense vs. Hybrid RRF):
```bash
python scripts/evaluate_hybrid_rrf.py
```
*Outputs comparative Hit@k, MRR@k, and NDCG@k metrics across benchmark queries.*

#### Answer Quality & Latency Benchmark:
```bash
python scripts/run_generation_evaluation.py
```
*Evaluates citation validity, excerpt faithfulness, abstention compliance, and latency across 28 queries; generates JSON and Markdown reports in `evaluation/results/`.*

---

## 6. Production Deployment Architecture

```
                    PRODUCTION CONTAINER TOPOLOGY

             Internet / Client Traffic (HTTPS :443)
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Reverse Proxy (Nginx / Caddy)     │
            │  - SSL Termination & Compression    │
            │  - Static Asset Caching             │
            └──────────────┬────────────────┬─────┘
                           │                │
            /api/* traffic │                │ UI traffic (/)
                           ▼                ▼
            ┌───────────────────┐    ┌────────────────────┐
            │ FastAPI Service   │    │ Next.js Standalone │
            │ (Uvicorn / Gunicorn│   │ (Node.js Server)   │
            │ Port: 8000        │    │ Port: 3000         │
            └─────────┬─────────┘    └────────────────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ Persistent Volume │
            │ data/qdrant/      │
            │ data/chunked/     │
            └───────────────────┘
```

### 6.1 Docker Deployment Pattern

#### Backend Dockerfile (`Dockerfile.backend`):
```dockerfile
FROM python:3.11-slim AS backend
WORKDIR /app

# Install system dependencies for RocksDB and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile (`frontend/Dockerfile`):
```dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM node:20-alpine AS frontend-runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=frontend-builder /app/public ./public
COPY --from=frontend-builder /app/.next/standalone ./
COPY --from=frontend-builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

---

## 7. Troubleshooting & Common Operational Issues

### Issue 1: RocksDB File Lock Contention (`data/qdrant/.lock`)
- **Symptom:** `RuntimeError: Failed to acquire lock on 'data/qdrant/.lock'` or `IO error: lock hold by another process`.
- **Root Cause:** Qdrant embedded mode enforces single-process exclusive locking. Another process (e.g., a background FastAPI worker or a previously aborted CLI script) still holds the file lock.
- **Resolution:**
  1. Check for running Python processes:
     - On Windows: `Get-Process python*`
     - On Linux: `ps aux | grep python`
  2. Terminate stale processes:
     - On Windows: `Stop-Process -Name python -Force`
     - On Linux: `pkill -9 python`
  3. If no processes exist, remove the stale lockfile:
     ```powershell
     Remove-Item "data/qdrant/.lock" -Force
     ```

### Issue 2: PyTorch OpenMP Multiple Initialization Error (Windows)
- **Symptom:** `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`
- **Root Cause:** PyTorch and Hugging Face libraries loading multiple copies of the Intel OpenMP runtime DLL.
- **Resolution:** Set the OpenMP workaround environment variable prior to launching:
  ```powershell
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  ```
  *(Or add `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the entrypoint).*

### Issue 3: Ollama Connection Refused / 404
- **Symptom:** `LLMProviderError: Ollama daemon unreachable at http://localhost:11434`.
- **Root Cause:** Ollama service is not running or the required model (`gemma3:4b`) has not been pulled.
- **Resolution:**
  1. Ensure the Ollama daemon is active:
     ```bash
     ollama serve
     ```
  2. Verify or pull the required model:
     ```bash
     ollama list
     ollama pull gemma3:4b
     ```

### Issue 4: Next.js Frontend Unable to Connect to Backend
- **Symptom:** "Unable to connect to Research Engine API" banner displayed in the UI.
- **Root Cause:** Backend server is not running on port 8000 or CORS origin is mismatched.
- **Resolution:**
  1. Verify backend health: `curl http://localhost:8000/api/system/health`.
  2. Verify `frontend/.env.local` contains `NEXT_PUBLIC_API_URL=http://localhost:8000`.
  3. Restart Next.js dev server: `npm run dev`.

