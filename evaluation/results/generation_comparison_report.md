# Generation Pipeline Evaluation: Dense vs. Hybrid Retrieval Comparison

**Execution Timestamp:** 2026-10-20T05:37:36.148548
**Provider:** `mock` | **Model:** `mock-model` | **Mock Simulation:** `Yes`
**Benchmark Scope:** 28 Questions (18 Answerable, 10 Out-of-Scope)

> [!NOTE]
> **Provider & Evaluation Semantics Disclosure:**
> Both evaluation runs used the same `mock` provider (`mock-model`) and the same ground-truth benchmark questions.
> Retrieval metrics (Hit@k, MRR) measure candidate document location in the top-k rankings.
> Generation metrics measure citation validity, excerpt faithfulness, target-page provenance, evidence support, and lexical answer matching.

---

## 1. Side-by-Side Generation Metrics Comparison

| Metric | Dense Baseline | Hybrid Retrieval | Delta (±) | Semantic Meaning |
|---|:---:|:---:|:---:|---|
| **Out-of-Scope Abstention Accuracy** | **100.0%** (10 / 10) | **100.0%** (10 / 10) | +0.0% | Evaluated strictly on 10 OOS questions |
| **Answerable Completion Rate** | **44.4%** (8 / 18) | **44.4%** (8 / 18) | +0.0% | Fraction answering without abstention |
| **Target Provenance Citation Rate** | **44.4%** (8 / 18) | **44.4%** (8 / 18) | +0.0% | Citations cover target paper & page |
| **Evidence Claim Supported Rate** | **38.9%** (7 / 18) | **44.4%** (8 / 18) | +5.5% | Cited passage text contains expected claims |
| **Expected Answer Match Rate (Lexical)** | **38.9%** (7 / 18) | **44.4%** (8 / 18) | +5.5% | Deterministic token-recall heuristic |
| **Appropriate Outcome Rate (Strict)** | **60.7%** (17 / 28) | **64.3%** (18 / 28) | +3.6% | Strict: OOS abstention + verified answerable |
| **Unsupported Answer Risk Count** | **0 / 18** | **0 / 18** | +0 | Answering without verified target evidence |
| **Citation Validity Rate** | **100.0%** (11 / 11) | **100.0%** (11 / 11) | 0.0% | Valid chunk ID resolution |
| **Citation Excerpt Faithfulness** | **100.0%** (11 / 11) | **100.0%** (11 / 11) | 0.0% | Verbatim excerpt presence in source |
| **Cold-Start Latency** | 23019.1ms | 23221.4ms | +202.3ms | Initial model/index loading roundtrip (isolated clean subprocesses) |
| **Warm Mean Generation Latency** | 14.8ms | 19.4ms | +4.6ms | Steady-state average across benchmark queries |
| **Warm Median Generation Latency** | 14.8ms | 19.1ms | +4.4ms | Steady-state median across benchmark queries |

---

## 2. Key Differences and Analysis

### Retrieval Metrics vs. Generation Metrics
- In pure retrieval evaluation (Task 4.1), hybrid retrieval improved Hit@5 from 66.7% (12/18) to 83.3% (15/18), and Hit@3 from 55.6% (10/18) to 72.2% (13/18), and MRR@5 from 0.4167 to 0.4509, while Hit@1 shifted from 27.8% (5/18) to 22.2% (4/18) due to RRF candidate reordering.
- In generation evaluation under mock simulation, answer synthesis inspects retrieved context chunks deterministically, falling back to rank-1 preview when no specialized query pattern matches.
- Hybrid retrieval changed **Target Provenance Citation Rate** from 44.4% to 44.4%, directly impacting unsupported answer risk.

### Metric Denominators: Per-Citation Validity vs. Per-Question Success
- **Per-Citation Denominator (Total Emitted Citations):**
  - **Citation Validity Rate:** Calculated as `total_valid_citations / total_citations_emitted`. Checks whether every citation emitted points to an actual chunk ID present in the retrieved context window.
  - **Citation Excerpt Faithfulness Rate:** Calculated as `total_faithful_excerpts / total_citations_emitted`. Checks whether each cited excerpt is present verbatim within the text of its cited chunk.
- **Per-Question Denominators (Questions Evaluated):**
  - **Answerable Metrics (Denominator: 18 Answerable Questions):** Completion Rate, Target Provenance Citation Rate, Evidence Claim Supported Rate, and Expected Answer Match Rate are computed out of 18 answerable questions.
  - **Out-of-Scope Metrics (Denominator: 10 Out-of-Scope Questions):** Abstention accuracy is computed strictly out of the 10 out-of-scope questions.
  - **Appropriate Outcome Rate (Denominator: 28 Total Evaluated Questions):** Evaluated over all 28 benchmark questions (18 answerable + 10 out-of-scope).

### Cold-Start vs. Steady-State Warm Latency

#### Timer Definitions & Inclusions
- **Cold-Start Latency (23019.1ms vs. 23221.4ms):** Measures the one-time, end-to-end initialization duration from invoking `generate()` on a freshly initialized pipeline. This timer includes loading Python dynamic libraries (PyTorch C++ backend, `torch_cpu.dll`), deserializing SentenceTransformer model weights from disk into memory, opening persistent Qdrant vector storage, building and tokenizing the BM25 chunk corpus (for hybrid), and executing the initial warmup query.
- **Warm Mean Generation Latency (14.8ms vs. 19.4ms):** Measures steady-state roundtrip generation latency across the 28 benchmark queries executed after warmup, excluding one-time cold-start initialization artifacts.
- **Warm Median Generation Latency (14.8ms vs. 19.1ms):** Median steady-state roundtrip latency across the 28 benchmark queries, robust to single-query operating system scheduling jitter.

#### Fairness Audit: In-Process Ordering Bias vs. Isolated Execution
- **In-Process Ordering Bias:** When dense and hybrid evaluations run sequentially within the **same Python process**, the first mode (`dense`) absorbs the full ~26-second library loading cost (`torch`, `sentence_transformers`, `transformers`), while the second mode (`hybrid`) artificially benefits from pre-imported modules in `sys.modules`, reporting an artifactually lower cold-start (~4.3s) and falsely appearing ~24 seconds faster.
- **Architectural Reality:** Because hybrid retrieval incorporates dense retrieval plus BM25 indexing, its true cold-start initialization cannot be faster than dense retrieval under equivalent conditions.
- **Isolated Subprocess Measurements:** Executed in independent, isolated Python subprocesses with clean runtime environments:
  - **Dense Cold-Start:** 23019.1ms
  - **Hybrid Cold-Start:** 23221.4ms
  - **Cold-Start Delta:** +202.3ms
  - This reflects the true physical cost: hybrid retrieval requires an additional ~200-300ms to load and tokenize the BM25 chunk corpus on top of the SentenceTransformer initialization.

#### Steady-State Dual-Retrieval Overhead
- In warm steady-state operation, hybrid retrieval incurs approximately +4.6ms of additional roundtrip latency per query (19.4ms vs. 14.8ms) due to dual execution (dense embedding search + BM25 keyword scoring) and Reciprocal Rank Fusion (RRF).

### Target-Page Provenance vs. Evidence Claim Support
- While hybrid retrieval cites target pages for 8 questions, only 8 cited passages contain verified claim tokens.
- Physical page matching confirms provenance, but does not guarantee the passage establishes the question's factual claim.

### Lexical Matching vs. Semantic Correctness
- Lexical token overlap heuristic isolates informative non-stopword tokens after subtracting question framing.
- Under mock simulation, the system merely echoes chunk previews. Lexical match rate is 44.4%. Real reasoning and synthesis requires a live LLM provider.

### Mock Provider Limitations
- The mock provider verifies pipeline wiring, citation formatting, chunk isolation, and error boundaries.
- Automated benchmarks under mock simulation must not be mistaken for live generative reasoning quality.
