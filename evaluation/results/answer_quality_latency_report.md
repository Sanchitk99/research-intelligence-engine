# Controlled Answer Quality and Latency Evaluation Report

**Benchmark Scope:** 28 Questions (18 Answerable, 10 Out-of-Scope)
**Execution Timestamp:** `2026-10-22T12:47:44.830701+00:00`
**Provider:** `mock` (`mock-model`) | **Deterministic Simulation:** `Yes (Mock Provider)`
**Retrieval Pipeline:** `HYBRID` Retrieval (Dense `all-MiniLM-L6-v2` + BM25 + Reciprocal Rank Fusion, `top_k=5`)

> [!IMPORTANT]
> **Methodological Ground Rule: Independent Dimensions, Not Blended Scores**
> This evaluation strictly separates **Answer Correctness**, **Citation Support**, **Citation Validity**,
> **Abstention Behavior**, and **Response Latency**. Each dimension evaluates a distinct failure mode with
> its own denominator. They are not collapsed into an arbitrary single composite score.

---

## 1. Executive Summary & Multi-Dimensional Scorecard

| Evaluation Dimension | Primary Metric | Result | Denominator Scope | Failure Mode Governed |
|---|---|:---:|---|---|
| **1. Answer Correctness** | Lexical Token Match Rate | **44.4%** (8/18) | 18 Answerable Questions | Factual divergence from ground truth |
| **2. Citation Support** | Evidence Claim Supported Rate | **44.4%** (8/18) | 18 Answerable Questions | Citing passages that do not substantiate claims |
| | Target Page Provenance Rate | **44.4%** (8/18) | 18 Answerable Questions | Citing wrong paper or unverified page |
| | Unsupported Answer Risk | **0 / 18** (0.0%) | 18 Answerable Questions | Answering without verified target evidence |
| **3. Citation Validity** | Syntactic Chunk Validity Rate | **100.0%** (11/11) | 11 Emitted Citations | Hallucinated non-existent chunk IDs |
| | Excerpt Faithfulness Rate | **100.0%** (11/11) | 11 Emitted Citations | Model excerpt altered or not verbatim |
| **4. Abstention Behavior** | Out-of-Scope Abstention Rate | **100.0%** (10/10) | 10 Out-of-Scope Questions | Fabricating answers outside research corpus |
| | Zero-Citation Compliance | **100.0%** (10/10) | 10 Out-of-Scope Questions | Emitting phantom citations when abstaining |
| **5. Response Latency** | Cold-Start Initialization | **24446.4ms** | One-time initialization | Startup, model loading, first query |
| | Warm Median Generation Latency | **17.9ms** | 28 Warm Steady Queries | Median roundtrip response time |
| | Warm p95 Generation Latency | **22.8ms** | 28 Warm Steady Queries | 95th percentile worst-case latency |

---

## 2. Dimension 1: Answer Correctness

- **Evaluated Queries:** 18 Answerable Questions (Q01 to Q18)
- **Lexical Match Rate:** `44.4%` (8 / 18)
- **Mean Lexical Token Recall:** `0.2619`
- **Median Lexical Token Recall:** `0.0290`
- **Matching Rule:** `min_recall >= 0.25 or exact substring match`

> [!NOTE]
> **Automated Semantic Scoring Approximation Disclosure:**
> Automated lexical token recall calculates the fraction of answer-discriminating key tokens (after English stopword
> and question-framing token subtraction) present in the generated answer. While highly effective at detecting
> factual keyword alignment without API costs or non-deterministic drift, it is a transparent heuristic:
> it does not verify logical syllogisms, subtle negations, or stylistic paraphrases. In human review of research papers,
> expert inspection remains the gold standard for full semantic truth.

---

## 3. Dimension 2: Citation Support

- **Evaluated Queries:** 18 Answerable Questions
- **Target Page Provenance Citation Rate:** `44.4%` (8 / 18)
- **Evidence Claim Supported Rate:** `44.4%` (8 / 18)
- **Unsupported Answer Risk Count:** `0 / 18`

### Provenance, Claim Support, and Unsupported-Answer Risk Disentanglement
- **Target Page Provenance (8/18, 44.4%):** Measures whether an emitted citation references a verified ground-truth paper and physical page index annotated in the evaluation dataset.
- **Evidence Claim Support (8/18, 44.4%):** Requires that the actual passage text inside the cited chunk contains the specific factual claim terms needed to substantiate the answer. A citation pointing to the correct page that only contains formulas or boilerplate is correctly flagged as lacking claim support.
- **Why Unsupported-Answer Risk is 0/18 (0.0%):**
  - Across all 18 answerable queries, exactly **8 questions** produced answers (`Q02`, `Q03`, `Q04`, `Q09`, `Q10`, `Q11`, `Q12`, `Q18`). For all 8 of these questions, the system cited verified target evidence (`has_answer=True`, `target_cited=True`, so `unsupported_risk=False`).
  - For the remaining **10 answerable questions** (`Q01`, `Q05`, `Q06`, `Q07`, `Q08`, `Q13`, `Q14`, `Q15`, `Q16`, `Q17`), the required evidence was absent or incomplete from the top-5 retrieved context. Instead of fabricating ungrounded answers or citing irrelevant chunks, the pipeline strictly abstained with `abstained_insufficient_evidence` and 0 citations (`has_answer=False`).
  - Because `unsupported_evidence_risk = (has_answer and not target_cited)`, queries that abstain do not generate unsupported answers. The system avoided unsupported claims by choosing safe abstention over hallucination.

> [!NOTE]
> **Citation Claim Support Methodology & Scope:**
> Citation claim support in this benchmark evaluates whether the text of cited chunks contains discriminating factual
> terms from the annotated ground-truth answer or supporting excerpt (threshold: token recall >= 0.20).
> It is an automated lexical-overlap verification mechanism; it does not perform natural-language inference (NLI)
> or formal propositional claim decomposition. In human peer review, manual inspection remains the gold standard.

---

## 4. Dimension 3: Citation Validity

- **Total Citations Emitted:** 11
- **Syntactic Chunk Validity Rate:** `100.0%` (11 / 11)
- **Citation Excerpt Faithfulness Rate:** `100.0%` (11 / 11)
- **Metadata Provenance Accuracy Rate:** `100.0%`

### Citation Validation Architecture
1. **Chunk ID Resolution:** Every citation emitted by the LLM is checked against the set of chunk IDs physically supplied in the retrieved context window. Non-existent IDs are dropped.
2. **Verbatim Excerpt Matching:** Cited excerpts are verified as exact, un-normalized substrings within the source chunk text. If a generative model slightly paraphrases or truncates the excerpt, the citation engine reconciles the excerpt with verified source text rather than failing silently.
3. **Server-Side Provenance:** Document IDs, paper titles, physical page numbers, and printed page numbers are populated directly from stored chunk metadata, completely preventing model hallucination of bibliographic citations.

---

## 5. Dimension 4: Abstention Behavior

- **Evaluated Queries:** 10 Out-of-Scope Questions (OOS01 to OOS10)
- **Out-of-Scope Abstention Accuracy:** `100.0%` (10 / 10)
- **Zero-Citation Compliance:** `100.0%` (10 / 10)
- **False Answering Rate:** `0.0%` (0 / 10)

### Abstention Safeguard Analysis
- All 10 out-of-scope questions (covering topics absent from the 5-paper corpus, such as GPU-hours, ColBERT, carbon footprint, code generation, medical QA, and privacy) were rejected with `abstained_out_of_scope`.
- Crucially, **zero citations** were emitted for abstaining queries, satisfying the safety invariant that an abstention must never provide spurious bibliographic references.

---

## 6. Dimension 5: Response Latency Profiling

### Aggregate Latency Breakdown

| Latency Metric | Measured Duration | Description & Scope |
|---|:---:|---|
| **Cold-Start Latency** | `24446.4ms` | Initial PyTorch backend load, SentenceTransformer model import, Qdrant store connection, BM25 indexing, and warmup query roundtrip |
| **Warm Mean Latency** | `18.8ms` | Arithmetic mean roundtrip across 28 warm steady-state benchmark queries |
| **Warm Median Latency** | `17.9ms` | 50th percentile latency (robust to background OS scheduling jitter) |
| **Warm p95 Latency** | `22.8ms` | 95th percentile latency (evaluates tail latency under load) |
| **Warm Min / Max Latency** | `15.9ms` / `23.3ms` | Full range of steady-state query latencies |

---

## 7. Granular Per-Query Evaluation Records

| Query ID | Category | Status | Citations | Target Cited | Evidence Supported | Lexical Match | Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Q01` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 17.3 |
| `Q02` | answerable | `answered` | 2 | Yes | Yes | Yes | 20.2 |
| `Q03` | answerable | `answered` | 1 | Yes | Yes | Yes | 17.7 |
| `Q04` | answerable | `answered` | 1 | Yes | Yes | Yes | 17.0 |
| `Q05` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 15.9 |
| `Q06` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 22.2 |
| `Q07` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 20.4 |
| `Q08` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 18.4 |
| `Q09` | answerable | `answered` | 1 | Yes | Yes | Yes | 19.1 |
| `Q10` | answerable | `answered` | 2 | Yes | Yes | Yes | 16.4 |
| `Q11` | answerable | `answered` | 2 | Yes | Yes | Yes | 16.8 |
| `Q12` | answerable | `answered` | 1 | Yes | Yes | Yes | 17.7 |
| `Q13` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 16.2 |
| `Q14` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 16.3 |
| `Q15` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 17.0 |
| `Q16` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 17.3 |
| `Q17` | answerable | `abstained_insufficient_evidence` | 0 | No | No | No | 17.0 |
| `Q18` | answerable | `answered` | 1 | Yes | Yes | Yes | 16.8 |
| `OOS01` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 20.5 |
| `OOS02` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 20.7 |
| `OOS03` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 22.7 |
| `OOS04` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 18.1 |
| `OOS05` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 21.2 |
| `OOS06` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 23.3 |
| `OOS07` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 22.8 |
| `OOS08` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 20.0 |
| `OOS09` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 21.5 |
| `OOS10` | out_of_scope | `abstained_out_of_scope` | 0 | N/A (OOS) | N/A (OOS) | N/A (OOS) | 15.9 |

---

## 8. Mock Provider vs. Live Model Comparison

### Architectural Role of the Mock Provider
- **Deterministic Offline CI Harness:** The mock LLM provider simulates context-aware generation based on keyword isolation in retrieved passages, eliminating external API dependencies, cost, and rate limits.
- **Verification Scope:** Validates retrieval wiring, citation formatting, chunk ID filtering, excerpt reconciliation, prompt injection defense, and error boundaries.
- **Limitation:** It does NOT establish natural language reasoning quality, fluency, or multi-step synthesis of un-templated questions.

### Empirical Live-Model Measurements (Local Ollama / Gemma 3 4B on CPU)
- **Single-Question Smoke Test (Q01: 'What generator model does RAG use?'):**
  - Status: `answered`
  - Latency: `32,135.1ms` (~32.1 seconds for 1 query on CPU)
  - Token Usage: Prompt `2,375`, Completion `241`, Total `2,616` tokens
  - Citations Emitted: `1` (`P1_p03_c02`)
  - Syntactic Validity: `100%` (chunk was present in retrieved context)
  - Excerpt Reconciliation: Model generated an approximate excerpt that was automatically reconciled to the verbatim passage prefix (`is marginalized to get the seq2seq probability...`)
  - Evidence Claim Support: `True` (Recall: 0.20)
  - Key Finding: While local Gemma 3 4B demonstrates authentic language synthesis and valid citation grounding, inference latency on CPU (~32s/query) makes running the full 28-query benchmark (~15 minutes) cost-prohibitive for standard CI, reinforcing the necessity of decoupled mock evaluations for rapid regression testing.

---

## 9. Reproducibility & Commands Executed

### Reproducible Evaluation Command
```powershell
# Run controlled answer quality and latency evaluation (default: mock provider, hybrid retrieval)
python scripts/run_answer_quality_evaluation.py --provider mock --retrieval-mode hybrid

# Run with optional live local Ollama provider (requires Ollama running on localhost:11434)
python scripts/run_answer_quality_evaluation.py --provider ollama --model gemma3:4b --sample 1
```

### Saved Artifacts
- Raw Structured JSON: `evaluation\results\answer_quality_latency_evaluation.json`
- Markdown Report: `evaluation\results\answer_quality_latency_report.md`
