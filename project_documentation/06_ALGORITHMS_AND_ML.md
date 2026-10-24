# 06 — Algorithms & Machine Learning Deep Dive

## Executive Summary

The **Research Intelligence Engine (RIE)** integrates specialized Natural Language Processing (NLP) algorithms, dense neural embeddings, inverted lexical indices, rank fusion mathematics, and rule-based query intent routing. This document presents the mathematical formulations, algorithmic logic, hyperparameters, and engineering safeguards powering each stage of the retrieval and generation pipeline.

---

## 1. Pagination Resolution Algorithm

### 1.1 The Citation Discrepancy Problem
In academic literature, physical PDF page indices ($p_{\text{physical}} \in \{1, 2, \dots, N\}$) rarely match the printed journal/proceedings page numbers ($p_{\text{printed}}$). 
1. **Conference Proceedings:** Published papers (such as Lewis et al. at NeurIPS 2020) are bound within massive conference volumes, causing the first page of the paper to bear a printed number such as page 9459.
2. **ArXiv Preprints:** Preprints often contain unnumbered cover sheets, front matter, or no formal page numbers at all.
3. **The Provenance Rule:** Citing an ungrounded printed page hallucinated by an LLM destroys academic reproducibility.

### 1.2 Algorithmic Formulation (`PaginationResolver`)
The resolver operates with a strict 3-tier precedence hierarchy:

```
                      ┌─────────────────────────────────────────┐
                      │ Input: physical_page_index (0-based)    │
                      └────────────────────┬────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────┐
                        │ Tier 1: Explicit Dictionary Map?    │───[Yes]──► Return map[index]
                        └──────────────────┬──────────────────┘
                                           │ [No]
                        ┌──────────────────▼──────────────────┐
                        │ Tier 2: Explicit Numeric Offset?    │───[Yes]──► Return index + offset
                        └──────────────────┬──────────────────┘
                                           │ [No]
                        ┌──────────────────▼──────────────────┐
                        │ Tier 3: Parse Regex mapping_spec    │
                        └──────────────────┬──────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │ (Preprint check)                │ (Offset plus)                   │ (Offset minus)
         ▼                                 ▼                                 ▼
"preprint format" /             "printed_page_i_plus_(\d+)"       "printed_page_i_minus_(\d+)"
"no printed page numbers"                  │                                 │
         │                                 ▼                                 ▼
         ▼                          Return index + Δ                  Return max(1, index - Δ)
    Return None
```

- **Invariant:** If a paper has no verified mapping (e.g. arXiv preprints `P2`, `P3`, `P4`, `P5`), `printed_page_number` evaluates to `None`. The engine falls back transparently to physical page citations (e.g., `[P1, p. 3, Chunk P1_p03_c04]`).

---

## 2. Boundary-Preserving Sentence Chunking

### 2.1 The Chunking Dilemma
Naive character splitters (e.g., slicing every 1,000 characters) frequently bisect sentences, equations, or scientific terms (e.g., splitting `"all-MiniLM-"` from `"L6-v2"` or bisecting citations like `"[Lewis et al., 2020]"`). Conversely, naive sentence splitters break when encountering abbreviations common in research papers (`Fig. 1`, `e.g.`, `i.e.`, `et al.`, `Sec. 3`).

### 2.2 Algorithm Logic (`TextChunker`)

`TextChunker` implements a two-pass algorithm: **Atomic Unit Segmentation** followed by **Rolling Boundary Aggregation**.

#### Pass 1: Atomic Unit Segmentation
Text is segmented along paragraph breaks (`\n\n+`), line breaks (`\n`), and sentence boundaries (`(?<=[.?!])\s+`). Every split candidate is tested against a protected abbreviation set:

$$\mathcal{A} = \{\text{"fig"}, \text{"table"}, \text{"eq"}, \text{"e.g."}, \text{"i.e."}, \text{"et al."}, \text{"ref"}, \text{"sec"}, \text{"vol"}, \dots\}$$

In addition, single uppercase initials (e.g., `"P."` in `"P. Lewis"`) are protected using regex:
```python
if len(raw_word.rstrip(".")) == 1 and raw_word[0].isupper():
    # Protected: do not treat as sentence terminator
    return True
```

If an individual sentence or section heading exceeds the maximum chunk size ($S_{\max} = 1000$ characters), it is subdivided along whitespace word boundaries, with a hard character slice as a last resort.

#### Pass 2: Rolling Boundary Aggregation
Atomic units $u_1, u_2, \dots, u_m$ are accumulated into a chunk until adding $u_{j}$ would exceed $S_{\max}$:

$$\text{span\_len}(u_i, u_j) = \text{end\_idx}(u_j) - \text{start\_idx}(u_i) \le S_{\max}$$

When a chunk closes, the next chunk does not begin at $u_{j}$. Instead, a **step-back window** is computed to guarantee context overlap ($S_{\text{overlap}} = 150$ characters):

$$i_{\text{next}} = \min \left\{ k \in [i+1, j] \mid \text{end\_idx}(u_j) - \text{start\_idx}(u_k) \le S_{\text{overlap}} \right\}$$

This guarantees semantic continuity across boundaries without duplicating full paragraphs.

---

## 3. Dense Representation: Sentence-Transformers

### 3.1 Neural Embedding Architecture
RIE utilizes `sentence-transformers/all-MiniLM-L6-v2` for dense vector search:

| Property | Value | Architectural Significance |
| :--- | :--- | :--- |
| **Base Architecture** | 6-layer MiniLM (Distilled BERT) | Low parameter count (22.7M parameters) |
| **Hidden Dimension** | 384 dimensions | Fast dot product computation; small RAM footprint |
| **Max Sequence Length** | 256 tokens | Aligns with 1000-character chunk budget |
| **Pooling Strategy** | Mean Pooling | Average of contextualized token representations |
| **Batch Size** | 32 | Balanced memory usage on standard CPU environments |

### 3.2 L2 Normalization & Cosine Similarity
All embeddings are normalized to unit Euclidean length ($\|\mathbf{v}\|_2 = 1.0$) upon generation:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{384} v_i^2}}$$

Because all stored passage vectors $\hat{\mathbf{d}}$ and runtime query vectors $\hat{\mathbf{q}}$ lie on the 384-dimensional unit hypersphere $\mathbb{S}^{383}$, **Cosine Similarity simplifies directly to the Euclidean Dot Product**:

$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}} = \sum_{i=1}^{384} \hat{q}_i \hat{d}_i$$

This reduces vector distance calculation in Qdrant to single-instruction vector dot products (BLAS/SIMD accelerated).

---

## 4. Lexical Search: Okapi BM25

### 4.1 BM25 Scoring Formulation
Lexical keyword search (`BM25Searcher`) evaluates keyword relevance using the Okapi BM25 probabilistic ranking function:

$$\text{BM25}(D, Q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $Q = \{q_1, q_2, \dots, q_m\}$ is the tokenized research query.
- $f(q_i, D)$ is the term frequency of token $q_i$ within chunk $D$.
- $|D|$ is the length of chunk $D$ in tokens.
- $\text{avgdl}$ is the average token length across all 396 chunks in the corpus.
- $\text{IDF}(q_i)$ is the Robertson-Spärck Jones Inverse Document Frequency:

$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$

Where $N = 396$ (total chunks) and $n(q_i)$ is the number of chunks containing $q_i$.

### 4.2 Parameter Calibration

| Hyperparameter | Value | Calibration Rationale |
| :--- | :--- | :--- |
| **$k_1$ (Term Saturation)** | `1.5` | Standard IR benchmark; prevents repetitive mentions from disproportionately dominating the score. |
| **$b$ (Length Normalization)** | `0.75` | Moderates length penalty; ensures long technical explanations are not excessively penalized. |
| **Tokenization** | Alphanumeric Regex + Lowercase | Strips non-alphanumeric punctuation while preserving hyphenated compounds (`self-rag`). |

---

## 5. Reciprocal Rank Fusion (RRF)

### 5.1 Why Score Normalization Fails
Combining dense vector scores and BM25 scores via linear combination ($\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{bm25}}$) is inherently fragile:
- Dense cosine similarities fall strictly in $[-1.0, 1.0]$ (typically $[0.3, 0.8]$ in practice).
- BM25 scores are unbounded $[0, +\infty)$, heavily dependent on document length and term frequency.
- Min-Max scaling of BM25 is query-dependent and highly sensitive to outliers.

### 5.2 The RRF Formulation
RIE uses **Reciprocal Rank Fusion (RRF)** (Cormack, Clarke & Büttcher, 2009), a scale-invariant rank-based fusion algorithm:

$$\text{RRF}(d) = \sum_{m \in M} w_m \cdot \frac{1}{k + r_m(d)}$$

Where:
- $M = \{\text{dense}, \text{bm25}\}$ are the two retrieval subsystems.
- $w_{\text{dense}} = 1.0$ and $w_{\text{bm25}} = 1.0$ (equal weight).
- $k = 60$ is the ranking smoothing constant.
- $r_m(d) \in \{1, 2, \dots, 50\}$ is the 1-based ordinal rank of chunk $d$ in subsystem $m$.
- If a chunk appears in only one candidate list, it receives the reciprocal rank from that system alone.

```
Subsystem 1 (Dense k=50)     Subsystem 2 (BM25 k=50)
  Rank 1: P1_p01_c01           Rank 1: P2_p04_c02
  Rank 2: P2_p04_c02           Rank 2: P1_p01_c01
  Rank 3: P3_p02_c01           Rank 3: P4_p10_c03
           │                            │
           └─────────────┬──────────────┘
                         ▼
             Reciprocal Rank Fusion
        RRF(c) = 1/(60 + r_dense) + 1/(60 + r_bm25)

  P1_p01_c01: 1/(60 + 1) + 1/(60 + 2) = 0.01639 + 0.01613 = 0.03252 -> [RANK 1]
  P2_p04_c02: 1/(60 + 2) + 1/(60 + 1) = 0.01613 + 0.01639 = 0.03252 -> [RANK 2 (tie-break cid)]
  P3_p02_c01: 1/(60 + 3) + 0           = 0.01587 + 0       = 0.01587 -> [RANK 3]
```

### 5.3 Candidate Depth ($k_{\text{candidate}} = 50$)
Each subsystem retrieves 50 candidates before fusion. This deep candidate pool ensures that chunks with moderate scores in both systems (e.g., rank 15 in Dense and rank 18 in BM25) can accumulate sufficient reciprocal rank to surpass single-system outliers (e.g., rank 8 in Dense but absent from BM25).

---

## 6. Query Intent Routing & Diversification

### 6.1 The Monopolization Problem
In comparative research queries (e.g., *"How do RAG and REALM differ in their passage retrieval mechanisms?"*), standard retrieval often returns 5 chunks all belonging to the single paper that has the highest lexical term overlap (e.g., 5 chunks from `P1`), completely starving the context window of evidence from the second paper (`P3`).

### 6.2 Deterministic Intent Classifier (`QueryIntentRouter`)
To solve this without incurring LLM latency or cost, RIE features a rule-based query classifier:

```
                          Natural Language Query
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
   Comparative Phrasing Cues                 Domain System Entities
   - "compare", "contrast"                   - RAG: ["rag", "rag-token", ...]
   - "difference between"                    - DPR: ["dpr", "bi-encoder", ...]
   - "versus", "vs"                          - REALM: ["realm", "orqa", ...]
   - "differ from"                           - Self-RAG: ["self-rag", "isrel", ...]
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
       (Cues matched >= 1 AND Distinct System Families >= 2)?
                     │                      │
                   [Yes]                   [No]
                     ▼                      ▼
           QueryIntent.COMPARATIVE    QueryIntent.STANDARD
                     │                      │
                     ▼                      ▼
           max_chunks_per_doc = 2     max_chunks_per_doc = None
```

### 6.3 Two-Pass Diversification Algorithm
When `COMPARATIVE` intent is active:
1. **Pass 1 (Constrained Allocation):** Fused candidates are inspected in descending RRF score order. Chunks are accepted into the top-k window only if their source document has contributed $< 2$ chunks so far.
2. **Pass 2 (Graceful Backfill):** If the candidate pool does not contain enough multi-document chunks to fill $k = 5$, the remaining slots are filled with the highest-scoring unselected chunks, preventing incomplete context windows.

---

## 7. Optional Cross-Encoder Re-Ranking

### 7.1 Cross-Encoder vs. Bi-Encoder
- **Bi-Encoder (`all-MiniLM-L6-v2`):** Encodes query $Q$ and passage $D$ independently into vectors $\mathbf{q}, \mathbf{d}$. Fast ($O(1)$ dot product at query time), but lacks cross-attention between query tokens and document tokens.
- **Cross-Encoder (`ms-marco-MiniLM-L2-v2`):** Feeds the concatenated pair $([CLS], Q, [SEP], D, [SEP])$ through all transformer layers. Every query token attends directly to every passage token, producing a unified relevance probability $P(\text{Relevant} \mid Q, D)$.

### 7.2 Safety & Resource Guardrails (`src/retrieval/reranker.py`)
Because cross-encoders require substantial CPU memory and inference time:
1. **Opt-In Only:** Reranking is completely decoupled from the production RAG pipeline and disabled by default.
2. **RAM Safeguard:** Prior to loading weights, the engine queries Windows OS memory via `ctypes.windll.kernel32.GlobalMemoryStatusEx`. If available RAM $< 3072\text{ MB}$, the reranker aborts immediately with `RerankerResourceError`.
3. **Local Cache Verification:** The engine inspects `HUGGINGFACE_HUB_CACHE` before invocation. If weights are not cached locally, it aborts rather than triggering a blocking multi-hundred-megabyte network download.

---

## 8. Evaluation Metrics & Statistical Methods

### 8.1 Retrieval Evaluation Metrics
Retrieval performance is benchmarked across 18 grounded test queries with known gold passage targets:

| Metric | Formulation | Interpretation |
| :--- | :--- | :--- |
| **Hit@k** | $\frac{1}{\|Q\|} \sum_{q=1}^{\|Q\|} \mathbb{I}(\text{rank}_q \le k)$ | Fraction of queries where at least one target chunk appears in top $k$. |
| **MRR@k** | $\frac{1}{\|Q\|} \sum_{q=1}^{\|Q\|} \frac{1}{\text{rank}_q}$ (if $\text{rank}_q \le k$, else $0$) | Mean Reciprocal Rank; penalizes targets appearing lower in the ranking. |
| **NDCG@k** | $\frac{1}{\|Q\|} \sum_{q=1}^{\|Q\|} \frac{\text{DCG}_k(q)}{\text{IDCG}_k(q)}$ | Normalized Discounted Cumulative Gain accounting for position decay. |

*Empirical Results:*
- Dense Baseline: Hit@1 27.8%, Hit@3 55.6%, Hit@5 66.7%, MRR@5 0.4167.
- Hybrid RRF ($k=60$): Hit@1 22.2%, Hit@3 72.2% (+16.6%), Hit@5 83.3% (+16.6% absolute / +25% relative), MRR@5 0.4509.

### 8.2 Answer Quality & Hallucination Metrics
Evaluated across 28 queries (18 in-scope grounded queries, 10 out-of-scope adversarial queries):

| Dimension | Formula / Definition | Measured Score |
| :--- | :--- | :--- |
| **Unsupported-Answer Risk** | $\frac{\text{Queries answering without evidence}}{\text{Total Grounded Queries}}$ | **0.0% (0 / 18)** |
| **Out-of-Scope Abstention Rate** | $\frac{\text{Queries abstaining on unanswerable topics}}{\text{Total Out-of-Scope Queries}}$ | **100.0% (10 / 10)** |
| **Zero-Citation Abstention Compliance** | $\frac{\text{Abstained responses emitting zero citations}}{\text{Total Abstained Responses}}$ | **100.0% (10 / 10)** |
| **Syntactic Citation Validity** | $\frac{\text{Citations with valid doc\_id, page, chunk\_id}}{\text{Total Citations Emitted}}$ | **100.0% (11 / 11)** |
| **Excerpt Faithfulness Rate** | $\frac{\text{Citation excerpts found verbatim in cited chunk}}{\text{Total Citations Emitted}}$ | **100.0% (11 / 11)** |
| **Lexical Token Recall Match** | $\frac{\text{Answers with token overlap with reference}}{\text{Total Grounded Answers}}$ | **44.4% (8 / 18)** |

### 8.3 Latency Measurement Boundaries
- **Cold-Start Boundary:** Measures end-to-end initialization, including Python module imports, SentenceTransformer weight loading from disk to CPU memory, Qdrant client connection, and first-query inference. Measured at **24,446.4 ms** (~24.4s).
- **Warm Inference Boundary (Mock Provider):** Measures pipeline overhead (query embedding, Qdrant vector retrieval, BM25 scoring, RRF fusion, citation validation). Measured at **17.9 ms median** and **22.8 ms p95**.
- **Live LLM Inference Boundary (Gemma 3 4B on CPU):** Includes full autoregressive token generation across 2,616 prompt tokens. Measured at **32,135.1 ms** (~32.1s).

