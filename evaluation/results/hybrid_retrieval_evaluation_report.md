# Hybrid Retrieval Evaluation Report (Task 4.1)

**Execution Timestamp:** 2026-10-19T03:19:27.785837
**Retriever:** `HybridRetriever` | BM25Okapi + `all-MiniLM-L6-v2` | RRF k=60
**Corpus:** 5 research papers, 82 pages, 396 chunks in `data/qdrant/` (`research_corpus`)

---

## 1. Aggregate Benchmark Metrics

| Metric | Score | Description |
|---|---|---|
| **Evaluated Questions** | **18** / 18 | All answerable questions evaluated; 10 out-of-scope questions excluded |
| **Hit@1 (Page-Level Acc.)** | **22.2%** | Relevant target page chunk appears at rank 1 (4 / 18) |
| **Hit@3 (Page-Level Acc.)** | **72.2%** | Relevant target page chunk appears within top 3 (13 / 18) |
| **Hit@5 (Page-Level Acc.)** | **83.3%** | Relevant target page chunk appears within top 5 (15 / 18) |
| **Mean Reciprocal Rank (MRR@5)** | **0.4509** | Truncated reciprocal rank across all 18 questions (sum = 8.1167 / 18) |
| **Mean Target Page Recall@5** | **49.1%** | Average fraction of annotated target pages retrieved in top 5 |
| **Mean Document Coverage@5 (All)** | **55.7%** | Average fraction of required documents retrieved in top 5 across all 18 queries |

### Multi-Document Question Coverage (n = 8)

| Multi-Document Metric | Score | Explanation |
|---|---|---|
| **Partial Document Coverage (Any)** | **100.0%** | At least one required document retrieved in top 5 (8 / 8) |
| **Full Document Coverage (All)** | **0.0%** | ALL required documents retrieved in top 5 (0 / 8) |
| **Mean Document Coverage (Multi-Doc)** | **37.9%** | Average percentage of required documents retrieved among multi-paper queries |

> [!NOTE]
> **Metric Definition & Ground-Truth Caveat:**
> 1. **Hit@k vs Recall@k:** In accordance with Information Retrieval conventions for question answering, the primary metric is **Hit@k** (binary top-k retrieval accuracy). Traditional chunk-level recall is not computable without an exhaustive annotation of every sentence across all 396 corpus chunks.
> 2. **Page-Level Matching Limitation:** A chunk is classified as a match when its physical page matches an annotated target page. While this verifies that the retrieval system surfaced the correct source page, it represents an upper-bound proxy because not every chunk on an annotated page contains the verbatim answer.
> 3. **MRR@5 Derivation:** Because retrieval is evaluated at cutoff top-$k=5$, each question's reciprocal rank is $1 / \text{rank}$ for the first target-page chunk found in ranks 1–5, or $0.0$ if no target page appears within the top 5 results. Chunks beyond rank 5 are not evaluated and receive a reciprocal rank of $0.0$. Thus, this metric is mathematically truncated MRR@5, not full-ranking MRR. The exact sum of reciprocal ranks is $8.1167$, yielding $\text{MRR@5} = 8.1167 / 18 = 0.4509$.

---

## 2. Performance Breakdown by Question Type

| Question Type | Count | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|
| **Comparison** | 4 | 25.0% | 75.0% | 100.0% | 0.5083 |
| **Factual** | 7 | 14.3% | 71.4% | 71.4% | 0.3810 |
| **Limitations** | 3 | 33.3% | 33.3% | 66.7% | 0.4167 |
| **Multi-paper** | 4 | 25.0% | 100.0% | 100.0% | 0.5417 |

---

## 3. Per-Question Evaluation Diagnostics

| ID | Type | Expected Targets | Top-1 Chunk (Score) | 1st Rel Rank | Reciprocal Rank | Hit@1 | Hit@3 | Hit@5 | Doc Cov. | All Docs? | Page Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Q01** | F | `P1:p[3]` | `P1_p04_c01 (0.032)` | 2 | 0.5000 | ❌ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q02** | F | `P2:p[3, 5, 6]` | `P2_p03_c04 (0.032)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 100% | ✅ | 67% |
| **Q03** | F | `P1:p[3]` | `P1_p04_c01 (0.033)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q04** | F | `P4:p[4]` | `P4_p17_c01 (0.031)` | 2 | 0.5000 | ❌ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q05** | F | `P5:p[3]` | `P5_p05_c02 (0.033)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q06** | M | `P1:p[5, 6], P2:p[8]` | `P1_p04_c05 (0.032)` | 2 | 0.5000 | ❌ | ✅ | ✅ | 50% | ❌ | 67% |
| **Q07** | M | `P1:p[6], P2:p[8], P3:p[7]` | `P1_p04_c05 (0.032)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 33% | ❌ | 33% |
| **Q08** | C | `P1:p[4], P2:p[3], P3:p[4, 5]` | `P3_p05_c01 (0.033)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 33% | ❌ | 50% |
| **Q09** | C | `P1:p[2, 3], P4:p[4]` | `P4_p20_c05 (0.029)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 50% | ❌ | 33% |
| **Q10** | C | `P4:p[4], P5:p[3]` | `P4_p10_c04 (0.032)` | 5 | 0.2000 | ❌ | ❌ | ✅ | 50% | ❌ | 50% |
| **Q11** | F | `P2:p[5, 8]` | `P2_p08_c02 (0.032)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q12** | F | `P3:p[5, 7]` | `P3_p08_c02 (0.033)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 100% | ✅ | 50% |
| **Q13** | L | `P5:p[3, 4, 5]` | `P5_p01_c02 (0.029)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q14** | M | `P1:p[3], P2:p[3, 6], P3:p[5, 7], P4:p[4, 5, 6, 7, 8], P5:p[3]` | `P1_p14_c03 (0.030)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 20% | ❌ | 9% |
| **Q15** | F | `P4:p[5]` | `P4_p09_c02 (0.032)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q16** | C | `P1:p[4], P2:p[4], P3:p[6]` | `P3_p12_c01 (0.032)` | 2 | 0.5000 | ❌ | ✅ | ✅ | 33% | ❌ | 33% |
| **Q17** | M | `P1:p[6], P2:p[5, 8], P3:p[7]` | `P3_p07_c01 (0.030)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 33% | ❌ | 25% |
| **Q18** | L | `P1:p[4, 6, 19]` | `P1_p17_c03 (0.031)` | 4 | 0.2500 | ❌ | ❌ | ✅ | 100% | ✅ | 67% |

---

## 4. Analysis of Missed Evidence (Absent from Top-5)

The following 3 questions did not contain ground-truth target evidence in the top 5:

### Question Q05: "How does RAGAS compute its faithfulness metric?"
- **Expected Targets:** {'P5': [3]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.0325] `P5_p05_c02` (P5, Page 5): "annotators. We report the results in terms of ac- curacy (i.e. the fraction of instances on which the model agrees with the annotators). To put the results i..."
  - Rank 2 [0.0306] `P5_p05_c04` (P5, Page 5): "metrics are much closer aligned with the human judgements than the predictions from the two base- lines. For faithfulness, the Ragas prediction are in genera..."
  - Rank 3 [0.0290] `P5_p01_c05` (P5, Page 1): "In practice, RAG systems are often evaluated in terms of the language modelling task itself, i.e. by mea- suring perplexity on some reference corpus. How- ev..."
  - Rank 4 [0.0288] `P5_p05_c03` (P5, Page 5): "score for faithfulness in the range 0-10. context: [context] answer: [answer] Ties, where the same score is assigned by the LLM to both answer candidates, we..."
  - Rank 5 [0.0280] `P5_p05_c01` (P5, Page 5): "Faith. Ans. Rel. Cont. Rel. Ragas 0.95 0.78 0.70 GPT Score 0.72 0.52 0.63 GPT Ranking 0.54 0.40 0.52 Table 1: Agreement with human annotators in pairwise com..."

### Question Q13: "What are the known limitations of using LLMs as evaluators in RAGAS?"
- **Expected Targets:** {'P5': [3, 4, 5]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.0290] `P5_p01_c02` (P5, Page 1): "consider: the ability of the retrieval system to identify relevant and focused context passages, the ability of the LLM to exploit such passages in a faithfu..."
  - Rank 2 [0.0290] `P5_p01_c01` (P5, Page 1): "Ragas: Automated Evaluation of Retrieval Augmented Generation Shahul Es†, Jithin James†, Luis Espinosa-Anke∗♢, Steven Schockaert∗ †Exploding Gradients ∗Cardi..."
  - Rank 3 [0.0284] `P5_p01_c05` (P5, Page 1): "In practice, RAG systems are often evaluated in terms of the language modelling task itself, i.e. by mea- suring perplexity on some reference corpus. How- ev..."
  - Rank 4 [0.0279] `P5_p01_c03` (P5, Page 1): "firmly established with the introduction of ever larger LMs (Roberts et al., 2020). While the most recent Large Language Models (LLMs) capture enough knowled..."
  - Rank 5 [0.0277] `P4_p10_c05` (P4, Page 10): "consistent with Menick et al. (2022). Human annotators also find ISREL and ISSUP reflection token predictions are mostly aligned with their assessments. Appe..."

### Question Q15: "How does Self-RAG's critic model get its training data?"
- **Expected Targets:** {'P4': [5]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.0318] `P4_p09_c02` (P4, Page 9): "precisely grounded yet shorter outputs. Llama2-FT7B, which is the baseline LM trained on the same instruction-output pairs as SELF-RAG without retrieval or s..."
  - Rank 2 [0.0317] `P4_p04_c04` (P4, Page 4): "vocabulary plus reflection tokens). Specifically, we train the generator model M on a curated corpus with interleaving passages retrieved by a retriever R an..."
  - Rank 3 [0.0313] `P4_p02_c03` (P4, Page 2): "SELF-RAG learns to retrieve, critique, and generate text passages to enhance overall generation quality, factuality, and verifiability. consistently retrieve..."
  - Rank 4 [0.0305] `P4_p10_c03` (P4, Page 10): "150k training instances, and fine-tune four SELF-RAG 7B variants on those subsets. Then, we compare the model performance on PopQA, PubHealth, and ASQA (cita..."
  - Rank 5 [0.0296] `P4_p06_c01` (P4, Page 6): "Preprint. separate reward models during training, we compute critique offline and directly insert them into the training corpus, where the generator LM is tr..."
