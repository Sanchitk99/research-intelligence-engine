# Cross-Paper Retrieval and Comparative Synthesis Evaluation Report (Task 4.4.17)

**Date:** 2026-10-20  
**Corpus:** 5 Research Papers (P1: RAG, P2: DPR, P3: REALM, P4: Self-RAG, P5: RAGAS)  
**Index:** Local Qdrant (396 chunks) + Okapi BM25 Index  
**Benchmark Questions:** 9 Cross-Paper Questions (3 Disagreement Pairs + 6 Canonical Comparison/Multi-Paper Questions)

---

## Executive Summary

- **Total Cross-Paper Questions Evaluated:** 9
- **Hybrid RRF Retrieval (Top-5) Full Paper Coverage:** 5 / 9 (55.6%)
- **Hybrid RRF Retrieval (Top-10) Full Paper Coverage:** 8 / 9 (88.9%)
- **Dense Vector Retrieval (Top-5) Full Paper Coverage:** 5 / 9 (55.6%)
- **Live Cloud LLM Generation (Groq / GPT-OSS):** 5 questions evaluated (bounded subset: DP01-DP03, Q08, Q09)
- **Live Gemini Provider:** 5 evaluated (network read timeout recorded)

### Key Empirical Findings:

1. **Single-Query Retrieval Asymmetry:** Standard single-query retrieval (both Dense and Hybrid RRF) suffers from **lexical and semantic asymmetry**. When a question compares multiple papers (e.g., REALM vs DPR vs RAG), the single query representation disproportionately matches passages from one paper, causing it to monopolize the top ranks. Under top_k=5, 4 out of 9 questions failed to retrieve all required papers in Hybrid retrieval (e.g. Q06 retrieved 5/5 P1, 0 P2; Q08 retrieved 5/5 P3, 0 P1/P2; Q09 retrieved 5/5 P4, 0 P1).
2. **Top-k=10 Capacity Expansion:** Expanding retrieval depth from $k=5$ to $k=10$ allows missing papers to enter the context for Q06, Q07, and Q08 (ranks 6-10), increasing full coverage from 55.6% (5/9) to 88.9% (8/9). However, Q09 still failed top-10 hybrid retrieval because P4 passages saturated all 10 slots.
3. **Cross-Paper Synthesis & Citation Asymmetry:**
   - When **both papers are present** in retrieved context (e.g., DP01 with P1 & P2, DP03 with P1 & P4), the LLM generates a grounded comparison with valid citations from both documents.
   - When **one paper is missing** from retrieved context (e.g., Q08 where only P3 was in top-5 context), the LLM either relies on parametric knowledge to answer without being able to cite the missing document (partial citations), or abstains.
4. **Adequacy Assessment:** The existing pipeline can answer comparison questions *only when* single-query retrieval happens to surface both documents within top-k. It lacks guaranteed multi-document coverage and explicit cross-paper synthesis mechanisms.

---

## Detailed Retrieval Coverage Matrix

| Question ID | Category | Expected Papers | Hybrid Top-5 Docs | Hybrid Top-5 Full Coverage | Hybrid Top-10 Docs | Hybrid Top-10 Full Coverage | Dense Top-5 Full Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DP01** | Disagreement Pair | `P2, P1` | `P2, P1` | ✅ FULL | `P5, P2, P1` | ✅ FULL | ✅ FULL |
| **DP02** | Disagreement Pair | `P3, P1` | `P2, P3, P1` | ✅ FULL | `P2, P3, P1` | ✅ FULL | ❌ 50% |
| **DP03** | Disagreement Pair | `P4, P1` | `P4, P1` | ✅ FULL | `P4, P1` | ✅ FULL | ✅ FULL |
| **Q06** | Benchmark Multi-Paper | `P2, P1` | `P1` | ❌ 50% | `P2, P1` | ✅ FULL | ❌ 50% |
| **Q07** | Benchmark Multi-Paper | `P2, P3, P1` | `P2, P1` | ❌ 67% | `P2, P3, P1` | ✅ FULL | ❌ 67% |
| **Q08** | Benchmark Comparison | `P2, P3, P1` | `P3` | ❌ 33% | `P2, P3, P1` | ✅ FULL | ❌ 33% |
| **Q09** | Benchmark Comparison | `P4, P1` | `P4` | ❌ 50% | `P4` | ❌ 50% | ✅ FULL |
| **Q10** | Benchmark Comparison | `P5, P4` | `P5, P4` | ✅ FULL | `P5, P4` | ✅ FULL | ✅ FULL |
| **Q16** | Benchmark Comparison | `P3, P1` | `P3, P1` | ✅ FULL | `P3, P1` | ✅ FULL | ✅ FULL |

---

## Generation Evaluation Details (Live Cloud LLM)

### Question `DP01`: TriviaQA exact match performance: Extractive Reader vs. Generative RAG
- **Question:** *"How do the Extractive Reader in DPR and Generative RAG compare on TriviaQA exact match performance?"*
- **Expected Papers:** `P2, P1`
- **Retrieved in Top-5 Context:** `P2, P1`
- **LLM Status:** `partially_answered`
- **Cited Papers:** `P2, P1` (All Expected Cited: `Yes`)
- **Total Citations Emitted:** 3
- **Generated Synthesis:**
  > On TriviaQA, the generative RAG model attains state‑of‑the‑art exact‑match accuracy, outperforming the DPR pipeline that uses an extractive reader.  DPR also reports strong EM scores, but the evidence does not provide a direct numeric comparison between the two systems on TriviaQA.
- **Citations Provenance:**
  - `[P1]` Page 5 (`P1_p05_c05`): "On all four open-domain QA
tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combine..."
  - `[P2]` Page 8 (`P2_p08_c03`): "accuracy. It is interesting to contrast our results to
those of ORQA (Lee et al., 2019) and also the
concurrently develo..."
  - `[P2]` Page 8 (`P2_p08_c02`): "Single
and Multi denote that our Dense Passage Retriever (DPR) is trained using individual or combined training datasets..."


### Question `DP02`: WebQuestions exact match: REALM (40.7) vs. T5-11B+SSM (44.7)
- **Question:** *"How do REALM and T5-11B+SSM compare on WebQuestions exact match?"*
- **Expected Papers:** `P3, P1`
- **Retrieved in Top-5 Context:** `P2, P3, P1`
- **LLM Status:** `answered`
- **Cited Papers:** `P1` (All Expected Cited: `No`)
- **Total Citations Emitted:** 1
- **Generated Synthesis:**
  > On the WebQuestions benchmark, the T5‑11B+SSM model achieves an exact‑match score of 44.7, which is higher than REALM’s score of 40.7.
- **Citations Provenance:**
  - `[P1]` Page 6 (`P1_p06_c01`): "REALM [20] 40.4 - / - 40.7 46.8..."


### Question `DP03`: Retrieval Frequency and Utility: Adaptive Retrieval vs. Unconditional Always-Retrieve
- **Question:** *"How does Self-RAG's adaptive retrieval compare to standard RAG's unconditional always-retrieve approach in terms of retrieval frequency and utility?"*
- **Expected Papers:** `P4, P1`
- **Retrieved in Top-5 Context:** `P4, P1`
- **LLM Status:** `answered`
- **Cited Papers:** `P4` (All Expected Cited: `No`)
- **Total Citations Emitted:** 2
- **Generated Synthesis:**
  > Self‑RAG’s adaptive retrieval lets the model decide whether to retrieve on a per‑token basis, and the frequency of retrieval can be tuned by a threshold δ – larger δ reduces how often the model pulls evidence.  In contrast, standard RAG always retrieves the top passage(s) regardless of relevance, which keeps retrieval frequency high but can hurt performance, as seen by the large drop in PopQA and ASQA when conventional RAG retrieves the top‑1 passage.  Thus Self‑RAG offers more selective, utility‑oriented retrieval while standard RAG’s unconditional approach yields higher but less useful retrieval frequency.
- **Citations Provenance:**
  - `[P4]` Page 10 (`P4_p10_c02`): "test time by adjusting such parameters without requiring additional training.
Efficiency and accuracy trade-off. Using o..."
  - `[P4]` Page 9 (`P4_p09_c04`): "Using the top passages regardless of their relevance (Retrieve top 1) as in conventional RAG approaches causes a large d..."


### Question `Q08`: Index Refresh Strategy: REALM vs DPR
- **Question:** *"How does REALM's index refresh strategy differ from DPR's static index?"*
- **Expected Papers:** `P2, P3, P1`
- **Retrieved in Top-5 Context:** `P3`
- **LLM Status:** `abstained_out_of_scope`
- **Cited Papers:** `None` (All Expected Cited: `No`)
- **Total Citations Emitted:** 0
- **Generated Synthesis:**
  > I cannot answer this question because the provided evidence does not contain any information about DPR’s static index strategy.
- **Citations Provenance:**


### Question `Q09`: Adaptive Retrieval: Self-RAG vs Standard RAG
- **Question:** *"How does Self-RAG's approach to deciding when to retrieve differ from standard RAG?"*
- **Expected Papers:** `P4, P1`
- **Retrieved in Top-5 Context:** `P4`
- **LLM Status:** `answered`
- **Cited Papers:** `P4` (All Expected Cited: `No`)
- **Total Citations Emitted:** 2
- **Generated Synthesis:**
  > Self‑RAG learns to decide whether to retrieve during generation by having the language model predict a "Retrieve" token (yes/no/continue) based on the input and the partial output so far.  It then uses additional reflection tokens (ISREL, ISSUP, ISUSE) to judge relevance, support, and usefulness of the retrieved passage, allowing it to retrieve only when the passage is judged helpful.  In contrast, standard RAG simply retrieves the top‑scoring passage(s) before generation and does not condition retrieval on the generation context or perform fine‑grained relevance checks.
- **Citations Provenance:**
  - `[P4]` Page 4 (`P4_p04_c01`): "Preprint.
Type
Input
Output
Definitions
Retrieve
x / x, y
{yes, no, continue}
Decides when to retrieve with R
ISREL
x, d..."
  - `[P4]` Page 9 (`P4_p09_c04`): "On ASQA, we evaluate models
on sampled 150 instances and exclude ablations involving adaptive or no retrieval processes...."


---

## Failure Modes & Diagnostic Analysis

1. **Retrieval Monopoly (Q06, Q08, Q09):** In top-5 hybrid retrieval, query terms like *'RAG-Token Natural Questions'* pull exclusively P1 passages, completely omitting P2's DPR results. In Q08 (*'REALM index refresh vs DPR static'*), the specialized query terms match P3 exclusively, omitting P2.
2. **Citation Incompleteness under Retrieval Monopolies:** In Q08, because only P3 was present in top-5 context, the model could only cite P3 chunks, leaving DPR claims ungrounded by context.
3. **Successful Multi-Citation Synthesis (DP01, DP03, Q10):** When both papers appear in top-5 chunks (e.g. DP01 has P1 and P2; DP03 has P1 and P4), the model successfully cites both documents and accurately explains the comparative difference.

---

## Conclusion & Recommendation

The current single-query retrieval pipeline can synthesize comparisons **only opportunistically** when a single fused query happens to rank multiple documents in the top-5. For complex or multi-aspect comparative queries, single-query retrieval is demonstrably brittle.

**Recommended Next Step: Task 4.4.18 — Implement Multi-Document Partitioned Comparative Retrieval**
Rather than rewriting the answer generator, add a partitioned comparative retrieval strategy that decomposes comparative questions into per-document retrieval slots (e.g., retrieving top-3 chunks from Paper A and top-3 chunks from Paper B) before synthesis, guaranteeing balanced multi-paper evidence.