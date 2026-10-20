# Retrieval Evaluation Baseline Report (Task 3.5 & Audit 3.5.1)

**Execution Timestamp:** 2026-10-19T03:19:21.450700
**Retriever:** `DenseRetriever` | **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, Cosine)
**Corpus:** 5 research papers, 82 pages, 396 chunks in `data/qdrant/` (`research_corpus`)

---

## 1. Aggregate Benchmark Metrics

| Metric | Score | Description |
|---|---|---|
| **Evaluated Questions** | **18** / 18 | All answerable questions evaluated; 10 out-of-scope questions excluded |
| **Hit@1 (Page-Level Acc.)** | **27.8%** | Relevant target page chunk appears at rank 1 (5 / 18) |
| **Hit@3 (Page-Level Acc.)** | **55.6%** | Relevant target page chunk appears within top 3 (10 / 18) |
| **Hit@5 (Page-Level Acc.)** | **66.7%** | Relevant target page chunk appears within top 5 (12 / 18) |
| **Mean Reciprocal Rank (MRR@5)** | **0.4167** | Truncated reciprocal rank across all 18 questions (sum = 7.5000 / 18) |
| **Mean Target Page Recall@5** | **42.1%** | Average fraction of annotated target pages retrieved in top 5 |
| **Mean Document Coverage@5 (All)** | **50.9%** | Average fraction of required documents retrieved in top 5 across all 18 queries |

### Multi-Document Question Coverage (n = 8)

| Multi-Document Metric | Score | Explanation |
|---|---|---|
| **Partial Document Coverage (Any)** | **62.5%** | At least one required document retrieved in top 5 (5 / 8) |
| **Full Document Coverage (All)** | **0.0%** | ALL required documents retrieved in top 5 (0 / 8) |
| **Mean Document Coverage (Multi-Doc)** | **27.1%** | Average percentage of required documents retrieved among multi-paper queries |

> [!NOTE]
> **Metric Definition & Ground-Truth Caveat:**
> 1. **Hit@k vs Recall@k:** In accordance with Information Retrieval conventions for question answering, the primary metric is **Hit@k** (binary top-k retrieval accuracy). Traditional chunk-level recall is not computable without an exhaustive annotation of every sentence across all 396 corpus chunks.
> 2. **Page-Level Matching Limitation:** A chunk is classified as a match when its physical page matches an annotated target page. While this verifies that the retrieval system surfaced the correct source page, it represents an upper-bound proxy because not every chunk on an annotated page contains the verbatim answer.
> 3. **MRR@5 Derivation:** Because retrieval is evaluated at cutoff top-$k=5$, each question's reciprocal rank is $1 / \text{rank}$ for the first target-page chunk found in ranks 1–5, or $0.0$ if no target page appears within the top 5 results. Chunks beyond rank 5 are not evaluated and receive a reciprocal rank of $0.0$. Thus, this metric is mathematically truncated MRR@5, not full-ranking MRR. The exact sum of reciprocal ranks is $7.5000$, yielding $\text{MRR@5} = 7.5000 / 18 = 0.4167$.

---

## 2. Performance Breakdown by Question Type

| Question Type | Count | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|---|
| **Comparison** | 4 | 25.0% | 50.0% | 75.0% | 0.3958 |
| **Factual** | 7 | 28.6% | 71.4% | 71.4% | 0.4524 |
| **Limitations** | 3 | 0.0% | 33.3% | 66.7% | 0.2500 |
| **Multi-paper** | 4 | 50.0% | 50.0% | 50.0% | 0.5000 |

---

## 3. Per-Question Evaluation Diagnostics

| ID | Type | Expected Targets | Top-1 Chunk (Score) | 1st Rel Rank | Reciprocal Rank | Hit@1 | Hit@3 | Hit@5 | Doc Cov. | All Docs? | Page Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Q01** | F | `P1:p[3]` | `P1_p02_c06 (0.647)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q02** | F | `P2:p[3, 5, 6]` | `P2_p13_c03 (0.489)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 100% | ✅ | 67% |
| **Q03** | F | `P1:p[3]` | `P1_p03_c03 (0.523)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q04** | F | `P4:p[4]` | `P4_p01_c05 (0.569)` | 2 | 0.5000 | ❌ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q05** | F | `P5:p[3]` | `P5_p03_c01 (0.446)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q06** | M | `P1:p[5, 6], P2:p[8]` | `P1_p06_c02 (0.557)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 50% | ❌ | 67% |
| **Q07** | M | `P1:p[6], P2:p[8], P3:p[7]` | `P1_p06_c02 (0.501)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 33% | ❌ | 33% |
| **Q08** | C | `P1:p[4], P2:p[3], P3:p[4, 5]` | `P3_p05_c01 (0.530)` | 1 | 1.0000 | ✅ | ✅ | ✅ | 33% | ❌ | 25% |
| **Q09** | C | `P1:p[2, 3], P4:p[4]` | `P4_p01_c05 (0.547)` | 4 | 0.2500 | ❌ | ❌ | ✅ | 50% | ❌ | 33% |
| **Q10** | C | `P4:p[4], P5:p[3]` | `P4_p01_c05 (0.497)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 50% | ❌ | 50% |
| **Q11** | F | `P2:p[5, 8]` | `P2_p06_c06 (0.643)` | 2 | 0.5000 | ❌ | ✅ | ✅ | 100% | ✅ | 100% |
| **Q12** | F | `P3:p[5, 7]` | `P3_p08_c02 (0.584)` | 3 | 0.3333 | ❌ | ✅ | ✅ | 100% | ✅ | 50% |
| **Q13** | L | `P5:p[3, 4, 5]` | `P1_p02_c06 (0.485)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q14** | M | `P1:p[3], P2:p[3, 6], P3:p[5, 7], P4:p[4, 5, 6, 7, 8], P5:p[3]` | `P3_p02_c07 (0.577)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q15** | F | `P4:p[5]` | `P4_p01_c05 (0.643)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q16** | C | `P1:p[4], P2:p[4], P3:p[6]` | `P3_p12_c01 (0.557)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q17** | M | `P1:p[6], P2:p[5, 8], P3:p[7]` | `P4_p17_c05 (0.488)` | — | 0.0000 | ❌ | ❌ | ❌ | 0% | ❌ | 0% |
| **Q18** | L | `P1:p[4, 6, 19]` | `P1_p17_c03 (0.551)` | 4 | 0.2500 | ❌ | ❌ | ✅ | 100% | ✅ | 33% |

---

## 4. Analysis of Missed Evidence (Absent from Top-5)

The following 6 questions did not contain ground-truth target evidence in the top 5:

### Question Q01: "What generator model does RAG use?"
- **Expected Targets:** {'P1': [3]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.6472] `P1_p02_c06` (P1, Page 2): "1Code to run experiments with RAG has been open-sourced as part of the HuggingFace Transform- ers Library [66] and can be found at https://github.com/hugging..."
  - Rank 2 [0.4294] `P1_p17_c03` (P1, Page 17): "We ﬁnd that doing Maximum Inner Product Search with FAISS is sufﬁciently fast on CPU, so we store document index vectors on CPU, requiring ∼100 GB of CPU mem..."
  - Rank 3 [0.4259] `P1_p04_c01` (P1, Page 4): "minimize the negative marginal log-likelihood of each target, P j −log p(yj|xj) using stochastic gradient descent with Adam [28]. Updating the document encod..."
  - Rank 4 [0.4220] `P4_p01_c05` (P4, Page 1): "If so, it outputs a retrieval token that calls a retriever model on demand (Step 1). Subsequently, SELF-RAG concurrently processes multiple retrieved passage..."
  - Rank 5 [0.4185] `P1_p08_c01` (P1, Page 8): "Table 4: Human assessments for the Jeopardy Question Generation Task. Factuality Speciﬁcity BART better 7.1% 16.8% RAG better 42.7% 37.4% Both good 11.7% 11...."

### Question Q13: "What are the known limitations of using LLMs as evaluators in RAGAS?"
- **Expected Targets:** {'P5': [3, 4, 5]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.4854] `P1_p02_c06` (P1, Page 2): "1Code to run experiments with RAG has been open-sourced as part of the HuggingFace Transform- ers Library [66] and can be found at https://github.com/hugging..."
  - Rank 2 [0.4135] `P5_p01_c04` (P5, Page 1): "sages from a corpus and feeding these passages, along with the original question, to the LM. While initial approaches relied on specialised LMs for retrieval..."
  - Rank 3 [0.4053] `P5_p01_c05` (P5, Page 1): "In practice, RAG systems are often evaluated in terms of the language modelling task itself, i.e. by mea- suring perplexity on some reference corpus. How- ev..."
  - Rank 4 [0.3860] `P4_p10_c03` (P4, Page 10): "150k training instances, and fine-tune four SELF-RAG 7B variants on those subsets. Then, we compare the model performance on PopQA, PubHealth, and ASQA (cita..."
  - Rank 5 [0.3611] `P1_p17_c03` (P1, Page 17): "We ﬁnd that doing Maximum Inner Product Search with FAISS is sufﬁciently fast on CPU, so we store document index vectors on CPU, requiring ∼100 GB of CPU mem..."

### Question Q14: "What base language models are used across the five papers?"
- **Expected Targets:** {'P1': [3], 'P2': [3, 6], 'P3': [5, 7], 'P4': [4, 5, 6, 7, 8], 'P5': [3]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.5771] `P3_p02_c07` (P3, Page 2): "While RC mod- 1We use the term “document” loosely to refer to a passage from the knowledge corpus, not necessarily a whole article. 2Strictly speaking, MLM i..."
  - Rank 2 [0.5583] `P1_p14_c02` (P1, Page 14): "URL https://www.aclweb. org/anthology/N19-4009. [46] Ethan Perez, Siddharth Karamcheti, Rob Fergus, Jason Weston, Douwe Kiela, and Kyunghyun Cho. Finding gen..."
  - Rank 3 [0.5357] `P5_p06_c06` (P5, Page 6): "jape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. Alex Mallen, Akari Asai, Victor Zh..."
  - Rank 4 [0.5344] `P4_p15_c04` (P4, Page 15): "Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. URL https://arxiv.org/abs/1909.08..."
  - Rank 5 [0.5232] `P1_p10_c05` (P1, Page 10): "https://www.aclweb.org/anthology/P17-1171. [6] Eunsol Choi, Daniel Hewlett, Jakob Uszkoreit, Illia Polosukhin, Alexandre Lacoste, and Jonathan Berant. Coarse..."

### Question Q15: "How does Self-RAG's critic model get its training data?"
- **Expected Targets:** {'P4': [5]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.6426] `P4_p01_c05` (P4, Page 1): "If so, it outputs a retrieval token that calls a retriever model on demand (Step 1). Subsequently, SELF-RAG concurrently processes multiple retrieved passage..."
  - Rank 2 [0.5690] `P4_p04_c05` (P4, Page 4): "3.2.1 TRAINING THE CRITIC MODEL Data collection for critic model. Manual annotation of reflection tokens for each segment is expensive (Wu et al., 2023). A s..."
  - Rank 3 [0.5597] `P4_p04_c04` (P4, Page 4): "vocabulary plus reflection tokens). Specifically, we train the generator model M on a curated corpus with interleaving passages retrieved by a retriever R an..."
  - Rank 4 [0.5514] `P4_p09_c02` (P4, Page 9): "precisely grounded yet shorter outputs. Llama2-FT7B, which is the baseline LM trained on the same instruction-output pairs as SELF-RAG without retrieval or s..."
  - Rank 5 [0.5420] `P4_p10_c03` (P4, Page 10): "150k training instances, and fine-tune four SELF-RAG 7B variants on those subsets. Then, we compare the model performance on PopQA, PubHealth, and ASQA (cita..."

### Question Q16: "Do RAG and REALM use the same knowledge corpus, and how do their passage sizes compare?"
- **Expected Targets:** {'P1': [4], 'P2': [4], 'P3': [6]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.5572] `P3_p12_c01` (P3, Page 12): "REALM: Retrieval-Augmented Language Model Pre-Training x: “Jennifer formed the production company Excellent Cadaver.” BERT also (0.13), then (0.08), later (0..."
  - Rank 2 [0.5433] `P3_p07_c05` (P3, Page 7): "The improvement of REALM over ORQA is purely due to bet- ter pre-training methods. The results also indicate that our method of pre-training can be applied b..."
  - Rank 3 [0.5266] `P1_p06_c02` (P1, Page 6): "Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, w..."
  - Rank 4 [0.5093] `P1_p10_c01` (P1, Page 10): "Broader Impact This work offers several positive societal beneﬁts over previous work: the fact that it is more strongly grounded in real factual knowledge (i..."
  - Rank 5 [0.5000] `P3_p02_c01` (P3, Page 2): "REALM: Retrieval-Augmented Language Model Pre-Training decide what knowledge to retrieve and use during inference. Before making each prediction, the languag..."

### Question Q17: "What evaluation benchmarks are shared across the 2020 papers, and which are unique to each?"
- **Expected Targets:** {'P1': [6], 'P2': [5, 8], 'P3': [7]}
- **Top-5 Retrieved Chunks:**
  - Rank 1 [0.4880] `P4_p17_c05` (P4, Page 17): "2019) and FEVER (Thorne et al., 2018) from the KILT benchmark (Petroni et al., 2021), ASQA (Stel- makh et al., 2022) and multiple QA datasets including ARC-E..."
  - Rank 2 [0.4759] `P3_p06_c06` (P3, Page 6): "baseline, we compare to concurrent work which ﬁne-tunes T5 for Open-QA (Roberts et al., 2020).4 We compare against the Base, Large, and even larger 11-billio..."
  - Rank 3 [0.4679] `P4_p08_c01` (P4, Page 8): "Preprint. Table 2: Overall experiment results on six tasks. Bold numbers indicate the best performance among non-proprietary models, and gray-colored bold te..."
  - Rank 4 [0.4617] `P2_p13_c04` (P2, Page 13): "better results compared to the usual retriever/reader training pipeline, resulting in the same 39.8 exact match score on NQ dev as in our regular reader mode..."
  - Rank 5 [0.4492] `P4_p07_c06` (P4, Page 7): "By default, we use the top five documents from Contriever-MS MARCO (Izacard et al., 2022a); for biographies and open-domain QA, we use additional top five do..."
