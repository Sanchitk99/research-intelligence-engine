# 10 — System Architecture Interview & Defense Guide

## Executive Summary

This document is engineered to prepare software engineers, machine learning engineers, and system architects to pitch, defend, and whiteboard the **Research Intelligence Engine (RIE)** in senior and staff-level technical interviews. It provides a crisp 2-minute elevator pitch, in-depth technical Q&A with empirical evidence, an architectural trade-off matrix, and a structured 45-minute whiteboarding guide.

---

## 1. The 2-Minute Architectural Pitch

> *"Most Retrieval-Augmented Generation (RAG) applications in production suffer from three critical flaws: first, dense semantic embeddings miss exact technical keywords, acronyms, and benchmark scores; second, character-based text splitters break sentences and scientific abbreviations mid-token; and third, LLMs regularly hallucinate non-existent citations and page numbers.*
> 
> *To solve this, I designed and built the **Research Intelligence Engine (RIE)**—a zero-external-infrastructure RAG platform built for academic literature review.
> 
> *Instead of relying solely on dense vector search, RIE implements a **hybrid retrieval pipeline** fusing 384-dimensional `all-MiniLM-L6-v2` dense vectors from an embedded RocksDB-backed **Qdrant** store with an **Okapi BM25** inverted lexical index using **Reciprocal Rank Fusion (RRF $k=60$)**. In our empirical benchmarks across research literature (including foundational papers on RAG, DPR, and REALM), hybrid RRF boosted **Hit@5 from 66.7% to 83.3%—a +25% relative improvement** over dense-only search.*
> 
> *On the ingestion side, RIE replaces naive character chunking with a **boundary-preserving sentence segmenter** that protects 25+ scientific abbreviations and single-letter author initials, decoupling physical PDF page indexes from printed journal pagination.*
> 
> *Finally, RIE enforces **strict citation provenance guardrails**: generated citations are verified against the retrieved context pool, hallucinated chunk IDs are stripped, quotation excerpts are checked for verbatim substring matches, and an automated rule-based **Query Intent Router** dynamically caps per-document chunks to ensure balanced representation in comparative queries.*
> 
> *The entire system runs on a modern stack: FastAPI backend, Next.js 15 App Router frontend with Tailwind CSS, and local or cloud LLM inference—delivering an audited **0.0% unsupported-answer risk** across our 28-query benchmark suite."*

---

## 2. Technical Q&A: Deep Dives & Grounded Answers

### Q1: Why did you choose Reciprocal Rank Fusion (RRF) instead of linear score combination (e.g., Min-Max or Softmax)?
**The Grounded Answer:**
> *"Linear score combination ($\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{bm25}}$) fails in production because dense cosine similarity and BM25 scores have fundamentally incompatible distributions. Cosine similarity is bounded in $[-1.0, 1.0]$ and typically clusters between $0.3$ and $0.8$ for natural language, whereas Okapi BM25 is unbounded $[0, +\infty)$ and varies wildly depending on query length and document frequency. 
> 
> Applying Min-Max scaling to BM25 makes the scores query-dependent and fragile to outliers. If one candidate has an unusually high BM25 score, all other candidates get compressed to near zero. 
> 
> **Reciprocal Rank Fusion (Cormack et al., 2009)** completely bypasses score calibration by operating solely on ordinal ranks:
> $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + r_m(d)}$$
> By setting $k = 60$, we smoothly penalize lower ranks without letting extreme score disparities in one system dominate the final ranking. In our benchmark, this rank-invariance directly produced a +16.6% absolute gain in Hit@3."*

---

### Q2: How does your text chunker prevent clipping critical scientific context?
**The Grounded Answer:**
> *"Generic chunkers (like LangChain's `RecursiveCharacterTextSplitter`) slice text based on raw character lengths or simple punctuation regex. In scientific papers, period characters appear constantly in abbreviations (`Fig. 1`, `e.g.`, `i.e.`, `et al.`, `Sec. 2.1`) and author names (`P. Lewis`). Naive splitters treat these as sentence terminators, isolating half a sentence into a separate chunk.
> 
> In `TextChunker`, we built a two-pass algorithm:
> 1. **Atomic Unit Segmentation:** We segment text on paragraph breaks and sentence boundary lookbehinds `(?<=[.?!])\s+`, but before splitting, we inspect the preceding token. We maintain a protected set of 25+ scientific abbreviations and check for single uppercase initials. If detected, the split is skipped.
> 2. **Rolling Boundary Aggregation with Step-Back:** Units are aggregated up to our 1,000-character budget. When closing a chunk, we compute an overlap step-back window ($S_{\text{overlap}} = 150$ characters) to find the first unit within the overlap distance. This guarantees semantic continuity without chopping phrases or repeating entire paragraphs."*

---

### Q3: How do you guarantee zero hallucinated citations? What happens if an LLM invents a chunk ID or misquotes a paper?
**The Grounded Answer:**
> *"We do not rely on prompt compliance alone. RIE implements a multi-layer post-generation citation validation pipeline in `src/generation/generator.py`:
> 1. **Context Pool Membership:** When the model emits a citation with `chunk_id`, we check whether that ID exists in the set of chunks passed into the prompt context for that specific query. If the model invents a chunk ID (`P1_p99_c99`), it is immediately rejected.
> 2. **Verbatim Excerpt Matching:** For accepted chunk IDs, we normalize the model's quoted excerpt (stripping punctuation and whitespace) and assert that it is a literal substring of the source chunk text. If an LLM paraphrased or fabricated a quote, we catch it and replace the excerpt with the verified passage prefix (`chunk.text[:120] + '...'`).
> 3. **Abstention Invariant:** If the model returns `abstained_out_of_scope` or `abstained_insufficient_evidence`, we enforce that zero citations are returned.
> 
> In our 28-query evaluation benchmark, this achieved **100% syntactic citation validity**, **100% excerpt faithfulness**, and **0.0% unsupported-answer risk**."*

---

### Q4: Why did you run Qdrant in embedded RocksDB mode rather than a hosted cloud vector database?
**The Grounded Answer:**
> *"The decision came down to **operational simplicity, data sovereignty, and latency overhead**:
> 1. **Zero External Dependencies:** Academic researchers and small teams often cannot provision managed cloud vector stores (like Pinecone) or maintain containerized infrastructure. Embedded Qdrant runs in-process via C++ RocksDB bindings.
> 2. **Sub-Millisecond Retrieval:** Because vectors are stored locally on the filesystem, we eliminate network roundtrips, TLS handshakes, and serialization over HTTP/gRPC.
> 3. **Deterministic Persistence:** Vectors and payloads are persisted directly on disk at `data/qdrant/`.
> 
> **The Trade-Off:** Embedded RocksDB creates an exclusive process lockfile (`.lock`), meaning only one OS process can access the database at a time. In our architecture roadmap, we designed an abstraction layer allowing the same `QdrantStoreManager` to connect to a remote Qdrant cluster (`http://localhost:6333`) when multi-worker scaling is required."*

---

### Q5: How does your Query Intent Router work without calling an LLM?
**The Grounded Answer:**
> *"Using an LLM to classify query intent adds 200–500 ms of latency and token costs to every request. 
> 
> Instead, we built `QueryIntentRouter` as a lightweight, deterministic rule-based engine:
> - It maintains two regex catalogs: **Comparative Phrasing Cues** (e.g. `compare`, `difference between`, `versus`, `how does X differ from Y`) and **Domain System Entities** (e.g. `RAG`, `DPR`, `REALM`, `Self-RAG`, `Ragas`).
> - It evaluates whether a query contains at least one comparative phrasing cue AND references at least two distinct system entity families.
> - If matched, it classifies the query as `COMPARATIVE` and triggers a **per-document chunk cap** (`max_chunks_per_doc = 2`).
> - This prevents a single dominant paper from monopolizing all 5 retrieved context slots, ensuring balanced comparative context with zero LLM overhead."*

---

## 3. Architectural Trade-Off Analysis

| Architectural Decision | Alternative Considered | Trade-Off & Justification |
| :--- | :--- | :--- |
| **Hybrid RRF ($k=60$)** | Dense-only search or BM25-only | Dense search alone missed exact keyword queries (Hit@5 was only 66.7%). Hybrid RRF lifted Hit@5 to 83.3% (+25% relative) by combining semantic recall with exact keyword precision. |
| **Embedded Qdrant (RocksDB)** | Cloud vector database (Pinecone, Weaviate) | Cloud databases require external credentials, internet connectivity, and network hops. Embedded RocksDB provides zero-config local persistence and sub-millisecond retrieval. |
| **Rule-Based Intent Router** | LLM-based query classifier | Calling an LLM to route queries introduces 300ms+ latency and API expense. Deterministic regex matching executes in $< 1\text{ ms}$ with zero runtime cost. |
| **Protected Sentence Chunker** | Naive fixed-size character chunking | Fixed character chunking cuts sentences and acronyms in half. Sentence boundary preservation with abbreviation protection ensures semantic coherence. |
| **Pydantic V2 Strict JSON Output** | Freeform Markdown text generation | Freeform prose cannot be reliably parsed or validated. Strict JSON schemas allow automated citation verification, excerpt verification, and status categorization. |

---

## 4. Whiteboard System Design Guide (45-Minute Interview)

When asked to *“Design a verified academic research Q&A system”*, follow this structured progression:

```
                                SYSTEM DESIGN PROGRESSION

   [00 - 05 min]   Clarify Requirements, Corpus Scale & Provenance SLA
         │
   [05 - 15 min]   High-Level Architecture & Storage Topology
         │
   [15 - 28 min]   Deep Dive: Ingestion, Sentence Chunking & Hybrid Retrieval
         │
   [28 - 38 min]   Deep Dive: Prompt Boundaries & Citation Provenance Engine
         │
   [38 - 45 min]   Scale, Bottlenecks & Operational Failure Modes
```

### Stage 1: Clarify Scope & Non-Functional Requirements (5 mins)
- *Scale:* 5 to 10,000 papers (~500,000 chunks).
- *Latency SLA:* $< 50\text{ ms}$ retrieval; $< 3\text{ s}$ generation (cloud LLM) or streaming local.
- *Factual Precision:* Zero tolerated hallucinated citations. Must abstain if evidence is missing.

### Stage 2: Draw High-Level Architecture (10 mins)
Draw the 3 core layers:
1. **Ingestion & Indexing Pipeline (Offline/Batch):** PDF $\rightarrow$ Page Extraction $\rightarrow$ Sentence Boundary Chunker $\rightarrow$ MiniLM Embeddings $\rightarrow$ Qdrant (Dense) + BM25 (Lexical).
2. **Retrieval & Fusion Layer (Online):** Query $\rightarrow$ Intent Router $\rightarrow$ Parallel Dense ($k=50$) & BM25 ($k=50$) $\rightarrow$ Reciprocal Rank Fusion ($k=60$) $\rightarrow$ Context Pool ($k=5$).
3. **Generation & Verification Layer:** Prompt Builder with XML Boundaries $\rightarrow$ LLM $\rightarrow$ Citation Validator $\rightarrow$ Structured Response.

### Stage 3: Retrieval & Fusion Deep Dive (13 mins)
- Explain why Dense alone fails on academic queries (acronyms, model names, author citations).
- Write the RRF formula on the whiteboard: $\text{RRF}(d) = \sum \frac{w_m}{k + r_m(d)}$.
- Explain why $k=60$ works (dampens top-rank disparity while rewarding consensus).

### Stage 4: Verification & Anti-Hallucination Engine (10 mins)
- Diagram the Citation Validator: `chunk_id` in context pool $\rightarrow$ excerpt in `chunk.text` $\rightarrow$ abstention zero-citation invariant.
- Explain categorical statuses (`answered`, `abstained_out_of_scope`, `abstained_insufficient_evidence`, `conflicting_evidence`).

### Stage 5: Scaling Bottlenecks & Failure Modes (7 mins)
- *What if the corpus grows to 10M chunks?* Transition Qdrant from embedded RocksDB to distributed cluster; partition by paper collection; replace in-memory BM25 with Elasticsearch/OpenSearch.
- *What if the server crashes during write?* Explain UUIDv5 idempotency and `.bak` atomic file operations.
- *How do you handle multi-user concurrency?* Replace RocksDB embedded engine with standalone Qdrant server mode over gRPC.

