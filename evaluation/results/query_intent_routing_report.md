# Query-Intent Routing Evaluation Report (Task 4.4.19)

**Date:** 2026-10-21 08:48 UTC  
**Corpus:** 5 Research Papers (P1: RAG, P2: DPR, P3: REALM, P4: Self-RAG, P5: RAGAS)  
**Evaluation Scope:** 37 Benchmark Queries (9 Cross-Paper, 18 Answerable, 10 Out-of-Scope)  
**Tested Strategies:** Always-Baseline vs. Always-Diversified vs. Intent-Routed  

---

## Executive Summary

1. **Intent Distribution:** Out of 37 total queries evaluated, **10 (27.0%)** were classified as `COMPARATIVE` (9 cross-paper questions + 1 out-of-scope comparative query OOS02) and **27 (73.0%)** were classified as `STANDARD`.
2. **Cross-Paper Multi-Document Coverage:**
   - **Always-Baseline:** 5 / 9 questions achieved full paper coverage (**55.6%**).
   - **Always-Diversified (`cap=2`):** 9 / 9 questions achieved full paper coverage (**100.0%**).
   - **Intent-Routed (Dynamic):** **9 / 9 questions achieved full paper coverage (100.0%)** — matching Always-Diversified.
3. **Single-Paper Recall Protection (18 Questions):**
   - Always-Diversified caused severe regression, dropping Hit@5 from **83.3% to 55.6%** (-27.8%) and MRR from 0.451 to 0.370.
   - **Intent-Routed** lifted Hit@5 to **72.2%** (+16.6% over Always-Diversified), successfully shielding factual single-paper questions (Q01–Q05, Q11–Q15, Q17–Q18) from premature chunk eviction.
4. **Abstention Safeguards:** On all 10 out-of-scope questions, Intent-Routed retrieval preserved a **100.0% abstention rate** (10/10), proving that routing comparative queries to diversified retrieval does not bypass abstention mechanisms.

---

## Comparative Benchmark Performance Matrix

| Strategy | Top-5 Cross-Paper Coverage | Single-Paper Hit@1 | Single-Paper Hit@3 | Single-Paper Hit@5 | Single-Paper MRR | OOS Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-Baseline** | 5/9 (55.6%) | 0.222 | 0.722 | 0.833 | 0.451 | 10/10 (100.0%) |
| **Always-Diversified** | **9/9 (100.0%)** | 0.222 | 0.556 | 0.556 (-0.278) | 0.370 | 10/10 (100.0%) |
| **Intent-Routed (Opt-In)** | **9/9 (100.0%)** | **0.222** | **0.667** | **0.722 (+0.166)** | **0.421** | **10/10 (100.0%)** |

---

## Detailed Intent Classification Breakdown

| ID | Category | Question Text | Classified Route | Cues Matched | Named Entities | Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **DP01** | Cross-Paper | How do the Extractive Reader in DPR and Generative RAG ... | `COMPARATIVE` | `compare` | `DPR, RAG` | Explicit comparative cue (compare) detected with multiple di... |
| **DP02** | Cross-Paper | How do REALM and T5-11B+SSM compare on WebQuestions exa... | `COMPARATIVE` | `compare` | `REALM, T5` | Explicit comparative cue (compare) detected with multiple di... |
| **DP03** | Cross-Paper | How does Self-RAG's adaptive retrieval compare to stand... | `COMPARATIVE` | `compare` | `RAG, Self-RAG` | Explicit comparative cue (compare) detected with multiple di... |
| **Q06** | Cross-Paper | What exact match scores does RAG-Token achieve on Natur... | `COMPARATIVE` | `compare` | `DPR, RAG` | Explicit comparative cue (compare) detected with multiple di... |
| **Q07** | Cross-Paper | How do RAG, DPR, and REALM compare on Natural Questions... | `COMPARATIVE` | `compare` | `DPR, RAG, REALM` | Explicit comparative cue (compare) detected with multiple di... |
| **Q08** | Cross-Paper | How does REALM's index refresh strategy differ from DPR... | `COMPARATIVE` | `how does realm's index refresh strategy differ` | `DPR, REALM` | Explicit comparative cue (how does realm's index refresh str... |
| **Q09** | Cross-Paper | How does Self-RAG's approach to deciding when to retrie... | `COMPARATIVE` | `how does self-rag's approach to deciding when to retrieve differ` | `RAG, Self-RAG` | Explicit comparative cue (how does self-rag's approach to de... |
| **Q10** | Cross-Paper | How does Self-RAG's ISSUP token relate to RAGAS's faith... | `COMPARATIVE` | `how does self-rag's issup token relate to` | `RAGAS, Self-RAG` | Explicit comparative cue (how does self-rag's issup token re... |
| **Q16** | Cross-Paper | Do RAG and REALM use the same knowledge corpus, and how... | `COMPARATIVE` | `compare` | `RAG, REALM` | Explicit comparative cue (compare) detected with multiple di... |
| **Q01** | Answerable Benchmark | What generator model does RAG use? | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |
| **Q02** | Answerable Benchmark | How does DPR train its passage encoder, and what types ... | `STANDARD` | `None` | `DPR` | Factual or multi-part query without explicit comparative int... |
| **Q03** | Answerable Benchmark | What is the difference between RAG-Sequence and RAG-Tok... | `STANDARD` | `difference between` | `RAG` | Comparative cue present (difference between), but all terms ... |
| **Q04** | Answerable Benchmark | What are the four reflection tokens in Self-RAG and wha... | `STANDARD` | `None` | `Self-RAG` | Factual or multi-part query without explicit comparative int... |
| **Q05** | Answerable Benchmark | How does RAGAS compute its faithfulness metric? | `STANDARD` | `None` | `RAGAS` | Factual or multi-part query without explicit comparative int... |
| **Q11** | Answerable Benchmark | On which dataset does BM25 outperform DPR, and why? | `STANDARD` | `None` | `BM25, DPR` | Factual or multi-part query without explicit comparative int... |
| **Q12** | Answerable Benchmark | What role does salient span masking play in REALM, and ... | `STANDARD` | `None` | `REALM, T5` | Factual or multi-part query without explicit comparative int... |
| **Q13** | Answerable Benchmark | What are the known limitations of using LLMs as evaluat... | `STANDARD` | `None` | `RAGAS` | Factual or multi-part query without explicit comparative int... |
| **Q14** | Answerable Benchmark | What base language models are used across the five pape... | `STANDARD` | `None` | `None` | Factual or multi-part query without explicit comparative int... |
| **Q15** | Answerable Benchmark | How does Self-RAG's critic model get its training data? | `STANDARD` | `None` | `Self-RAG` | Factual or multi-part query without explicit comparative int... |
| **Q17** | Answerable Benchmark | What evaluation benchmarks are shared across the 2020 p... | `STANDARD` | `None` | `None` | Factual or multi-part query without explicit comparative int... |
| **Q18** | Answerable Benchmark | What limitations does the RAG paper acknowledge about i... | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |
| **OOS01** | Out-of-Scope | What is the computational cost in GPU-hours of training... | `STANDARD` | `None` | `Self-RAG` | Factual or multi-part query without explicit comparative int... |
| **OOS02** | Out-of-Scope | How does ColBERT's late interaction retrieval compare t... | `COMPARATIVE` | `compare` | `ColBERT, DPR` | Explicit comparative cue (compare) detected with multiple di... |
| **OOS03** | Out-of-Scope | What is the carbon footprint of running RAGAS evaluatio... | `STANDARD` | `None` | `RAGAS` | Factual or multi-part query without explicit comparative int... |
| **OOS04** | Out-of-Scope | How do RAG systems perform on non-English languages? | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |
| **OOS05** | Out-of-Scope | What is the effect of using instruction-tuned models as... | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |
| **OOS06** | Out-of-Scope | How does retrieval-augmented generation perform on code... | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |
| **OOS07** | Out-of-Scope | What privacy risks arise from retrieval-augmented syste... | `STANDARD` | `None` | `None` | Factual or multi-part query without explicit comparative int... |
| **OOS08** | Out-of-Scope | How do RAGAS metrics perform when applied to medical or... | `STANDARD` | `None` | `RAG, RAGAS` | Factual or multi-part query without explicit comparative int... |
| **OOS09** | Out-of-Scope | What is the optimal chunk size for indexing research pa... | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |
| **OOS10** | Out-of-Scope | How do RAG systems handle contradictory evidence across... | `STANDARD` | `None` | `RAG` | Factual or multi-part query without explicit comparative int... |

---

## False-Positive & False-Negative Boundary Analysis

1. **Intra-Paper vs. Inter-Paper Comparison (Q03):**
   - Query: *'What is the difference between RAG-Sequence and RAG-Token?'*
   - Surface syntax contains `difference between`, but both models belong to a single paper family (`RAG`).
   - **Router Decision:** Conservatively classified as `STANDARD`.
   - **Outcome:** **Prevented a false positive.** Routing Q03 to `STANDARD` preserved all 5 slots for P1, successfully retrieving target page 3 at rank 3.
2. **Multi-Topic Factual Questions (Q02, Q04, Q11, Q12, Q17):**
   - Queries containing 'and' or multi-part factual questions were correctly classified as `STANDARD` without false triggering.
3. **Cross-Paper Questions in the Answerable Benchmark (Q09, Q10):**
   - Questions Q09 (Self-RAG vs RAG) and Q10 (Self-RAG vs RAGAS) are labeled as multi-paper in the original dataset.
   - The router correctly classified both as `COMPARATIVE` (retrieving both required papers with 100% full coverage), but because the 18-question answerable benchmark uses single-target page annotations, Hit@5 reflects 0.722 instead of 0.833.
4. **Out-of-Scope Comparative Query (OOS02):**
   - Query: *'How does ColBERT's late interaction retrieval compare to DPR's bi-encoder approach?'*
   - Correctly classified as `COMPARATIVE` (`ColBERT` vs `DPR`).
   - The generator verified that ColBERT was missing from evidence and properly issued `ABSTAINED_INSUFFICIENT_EVIDENCE`. Routing did not compromise safety.

---

## Known Limitations & Concrete Recommendation

1. **Known Limitations:**
   - The deterministic rule set relies on explicit comparison cues and domain entity recognition. Novel terminology or papers not in the domain dictionary without explicit comparison phrasing default to `STANDARD`.
   - Multi-paper questions with single-paper ground-truth benchmarks exhibit a trade-off between cross-paper coverage and single-target page recall.

2. **Concrete Recommendation:**
   **Keep Query-Intent Routing Opt-In (`enable_intent_routing=False` by default).**
   - For general single-paper workloads, unconstrained Hybrid RRF remains the safest baseline (Hit@5 = 83.3%).
   - When comparative synthesis is explicitly requested or when running cross-paper analysis, users or API callers can enable `enable_intent_routing=True`, boosting full comparative coverage from **55.6% to 100.0%** without risking the severe degradation of blind diversification.
