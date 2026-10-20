# Dense vs. Hybrid Retrieval Evaluation Report (Task 4.1)

**Execution Timestamp:** 2026-10-19T03:19:27.785837
**Baseline:** `DenseRetriever` (`all-MiniLM-L6-v2`, Cosine, top-k=5)
**Hybrid System:** `HybridRetriever` (BM25Okapi + `all-MiniLM-L6-v2`, RRF k=60, candidate_k=50)
**Corpus:** 5 research papers, 82 pages, 396 chunks in `data/qdrant/` (`research_corpus`)

---

## 1. Aggregate Performance Comparison

| Metric | Dense Baseline | Hybrid (Dense + BM25) | Delta | Relative Change |
|---|---|---|---|---|
| **Hit@1 (Page-Level Acc.)** | 27.8% | **22.2%** | 🔻 -5.6% | -20.0% |
| **Hit@3 (Page-Level Acc.)** | 55.6% | **72.2%** | 🔺 +16.7% | +30.0% |
| **Hit@5 (Page-Level Acc.)** | 66.7% | **83.3%** | 🔺 +16.7% | +25.0% |
| **Mean Reciprocal Rank (MRR@5)** | 0.4167 | **0.4509** | 🔺 +0.0342 | +8.2% |
| **Mean Target Page Recall@5** | 42.1% | **49.1%** | 🔺 +7.0% | +16.6% |
| **Mean Document Coverage@5** | 50.9% | **55.7%** | 🔺 +4.8% | +9.4% |
| **Multi-Doc Any Coverage@5** | 62.5% | **100.0%** | 🔺 +37.5% | +60.0% |
| **Multi-Doc Full Coverage@5** | 0.0% | **0.0%** | ➖ +0.0% | 0.0% |

---

## 2. Analysis of Previously Difficult Questions

Six questions failed to retrieve ground-truth evidence in top 5 under Dense retrieval:
`Q01`, `Q13`, `Q14`, `Q15`, `Q16`, and `Q17`.

| ID | Question | Expected Targets | Dense Rank | Hybrid Rank | Outcome | Key Factor |
|---|---|---|:---:|:---:|:---:|---|
| **Q01** | What generator model does RAG use?... | `P1:p[3]` | Miss | 2 | ✅ Rescued (Hit) | BM25 matched 'generator', 'model', 'RAG' with §2.3 BART passage |
| **Q13** | What are the known limitations of using LLMs ... | `P5:p[3, 4, 5]` | Miss | Miss | ➖ Unchanged | BM25 retrieved P5 evaluators/limitations context |
| **Q14** | What base language models are used across the... | `P1:p[3], P2:p[3, 6], P3:p[5, 7], P4:p[4, 5, 6, 7, 8], P5:p[3]` | Miss | 3 | ✅ Rescued (Hit) | Exact paper references matched across multi-paper query |
| **Q15** | How does Self-RAG's critic model get its trai... | `P4:p[5]` | Miss | Miss | ➖ Unchanged | Dense & BM25 balance for GPT-4 demonstration terms |
| **Q16** | Do RAG and REALM use the same knowledge corpu... | `P1:p[4], P2:p[4], P3:p[6]` | Miss | 2 | ✅ Rescued (Hit) | Exact matching on 'passage sizes', '288 wordpieces', '100-word' |
| **Q17** | What evaluation benchmarks are shared across ... | `P1:p[6], P2:p[5, 8], P3:p[7]` | Miss | 1 | ✅ Rescued (Hit) | Exact benchmark names (NQ, WQ, CT) matched in tables |

---

## 3. Per-Question Side-by-Side Diagnostics

| ID | Category | Expected | Dense Rank | Hybrid Rank | MRR@5 Delta | Status | Top Hybrid Chunk |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **Q01** | F | `P1:p[3]` | — | 2 | +0.5000 | ✅ Rescued | `P1_p04_c01 (RRF: 0.03200)` |
| **Q02** | F | `P2:p[3, 5, 6]` | 3 | 1 | +0.6667 | ⬆️ Improved | `P2_p03_c04 (RRF: 0.03200)` |
| **Q03** | F | `P1:p[3]` | 1 | 3 | -0.6667 | 🔻 Dropped | `P1_p04_c01 (RRF: 0.03252)` |
| **Q04** | F | `P4:p[4]` | 2 | 2 | 0.0000 | ➖ Same | `P4_p17_c01 (RRF: 0.03150)` |
| **Q05** | F | `P5:p[3]` | 1 | — | -1.0000 | 🔻 Regressed (Miss) | `P5_p05_c02 (RRF: 0.03252)` |
| **Q06** | M | `P1:p[5, 6], P2:p[8]` | 1 | 2 | -0.5000 | 🔻 Dropped | `P1_p04_c05 (RRF: 0.03227)` |
| **Q07** | M | `P1:p[6], P2:p[8], P3:p[7]` | 1 | 3 | -0.6667 | 🔻 Dropped | `P1_p04_c05 (RRF: 0.03178)` |
| **Q08** | C | `P1:p[4], P2:p[3], P3:p[4, 5]` | 1 | 1 | 0.0000 | ➖ Same | `P3_p05_c01 (RRF: 0.03252)` |
| **Q09** | C | `P1:p[2, 3], P4:p[4]` | 4 | 3 | +0.0833 | ⬆️ Improved | `P4_p20_c05 (RRF: 0.02939)` |
| **Q10** | C | `P4:p[4], P5:p[3]` | 3 | 5 | -0.1333 | 🔻 Dropped | `P4_p10_c04 (RRF: 0.03178)` |
| **Q11** | F | `P2:p[5, 8]` | 2 | 1 | +0.5000 | ⬆️ Improved | `P2_p08_c02 (RRF: 0.03178)` |
| **Q12** | F | `P3:p[5, 7]` | 3 | 3 | 0.0000 | ➖ Same | `P3_p08_c02 (RRF: 0.03279)` |
| **Q13** | L | `P5:p[3, 4, 5]` | — | — | 0.0000 | ➖ Same | `P5_p01_c02 (RRF: 0.02896)` |
| **Q14** | M | `P1:p[3], P2:p[3, 6], P3:p[5, 7], P4:p[4, 5, 6, 7, 8], P5:p[3]` | — | 3 | +0.3333 | ✅ Rescued | `P1_p14_c03 (RRF: 0.03012)` |
| **Q15** | F | `P4:p[5]` | — | — | 0.0000 | ➖ Same | `P4_p09_c02 (RRF: 0.03175)` |
| **Q16** | C | `P1:p[4], P2:p[4], P3:p[6]` | — | 2 | +0.5000 | ✅ Rescued | `P3_p12_c01 (RRF: 0.03178)` |
| **Q17** | M | `P1:p[6], P2:p[5, 8], P3:p[7]` | — | 1 | +1.0000 | ✅ Rescued | `P3_p07_c01 (RRF: 0.03048)` |
| **Q18** | L | `P1:p[4, 6, 19]` | 4 | 4 | 0.0000 | ➖ Same | `P1_p17_c03 (RRF: 0.03110)` |

---

## 4. Key Engineering Insights & Methodological Observations

1. **Complementary Modalities:** Dense semantic retrieval excels at high-level thematic similarity, whereas BM25 excels at exact technical entities ('BART-large', '288 wordpieces', 'Table 1').
2. **Reciprocal Rank Fusion (RRF) Robustness:** RRF eliminates score scale mismatch without requiring calibrated score normalization. Chunks retrieved with high consensus by both dense and sparse models naturally dominate the top ranks.
3. **MRR@5 Truncation Definition:** Because evaluation retrieval evaluates only top-5 chunks, any target evidence beyond rank 5 receives a reciprocal rank of 0.0. The metric is strictly MRR@5. Full-ranking MRR would assign non-zero reciprocal rank (e.g. 1/12 for Q05 at rank 12).
4. **Strict Annotation Independence:** Retrieval rankings were computed strictly without using ground-truth labels. The evaluation framework was applied purely post-hoc.