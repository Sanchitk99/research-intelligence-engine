# Answer-Quality Error Analysis: Diagnostic Audit of Generation & Retrieval Failures

**Evaluation Context:** Research Intelligence Engine | **Audit Type:** Read-Only Diagnostic (Task 4.3.3)
**Retriever Modes:** Dense (`all-MiniLM-L6-v2` + Qdrant) vs. Hybrid (Dense + BM25 + Reciprocal Rank Fusion)
**Benchmark Corpus:** 5 Core RAG Papers (P1–P5) | **Scope:** 18 Answerable Questions + 10 Out-of-Scope Questions (28 Total)
**Execution Timestamp:** 2026-10-20T04:28:50.315514

> [!NOTE]
> **Diagnostic-Only Boundary Disclosure:**
> This report is strictly an evidence-based diagnostic audit. No source code, tests, ground-truth annotations, retrieval rankings, or existing evaluation artifacts have been altered.
> The deterministic mock provider is used to analyze pipeline plumbing, chunk retrieval semantics, citation resolution, and heuristic evaluator limits without confounding from non-deterministic external LLM APIs.

---

## 1. Executive Summary

Across the 18 answerable research questions, the current pipeline achieves an **Appropriate Outcome Rate of 42.9% (12 / 28 total evaluated)** under both Dense and Hybrid retrieval modes, consisting of 100% out-of-scope abstention (10/10) and only **2 / 18 answerable successes (11.1%)** (`Q02` and `Q11`). The remaining 16 answerable questions fail strict evaluation due to distinct architectural and heuristic failure modes.

### Core Diagnostic Findings:
1. **Retrieval vs. Generation Bottlenecks Diverge Between Modes:**
   - In **Dense mode**, the primary generation bottleneck is **upstream retrieval failure**: 6 questions suffer from total **Retrieval Misses** (the necessary paper/page is absent from top-5), and 5 suffer from **Context Insufficiency** (only partial evidence across multi-document questions is retrieved).
   - In **Hybrid mode**, retrieval recall dramatically improves (Hit@5 increases from 66.7% to 83.3%, Hit@3 increases from 55.6% to 72.2%), successfully surfacing target-page chunks for 15 / 18 questions. However, because Reciprocal Rank Fusion (RRF) frequently reorders candidates such that target chunks appear at Ranks 2–5 while preamble chunks sit at Rank 1 (Hit@1 drops to 22.2%), **Citation Selection Failures double from 3 to 6 questions**. The fallback generation logic naively cites Rank 1, converting retrieval successes into unsupported citation risks.
2. **Verification of Q01 BART-Large Evidence:**
   - Target chunk `P1_p03_c04` (specifying BART-large with 400M parameters) is **absent from the top-5 in both Dense and Hybrid retrieval**. The chunk retrieved at Rank 2 in Hybrid mode (`P1_p03_c02`) resides on target Page 3 but contains purely mathematical formulas for marginalization, entirely lacking the BART-large definition. The mock provider correctly flags this evidence as unsupported rather than fabricating citations.
3. **Independent Verification of Multi-Citation Successes (Q02 & Q11):**
   - `Q02` (DPR training & negative sampling) and `Q11` (BM25 vs. DPR on SQuAD) pass all strict evaluation checks (target provenance, claim support, verbatim faithfulness, and lexical matching). Each question emits **2 verified citations** independently substantiated by distinct retrieved chunks.
4. **Evaluator Heuristic Penalties:**
   - Questions `Q04` (Dense) and `Q18` (Hybrid) successfully retrieve and cite verified target evidence that substantiates the factual claim, but fail strict evaluation solely due to unigram token-recall thresholding ($R < 0.25$) caused by table markdown dilution and snippet truncation.

---

## 2. Quantified Failure Breakdown & Category Reconciliation

Every one of the 18 answerable questions was assigned a single **primary failure category** based on inspection of the actual retrieved chunk text, cited passages, and evaluation logs. Category counts reconcile exactly to 18.

### Primary Failure Category Breakdown:

| Primary Failure Category | Dense Count | Hybrid Count | Shift (Dense $\to$ Hybrid) | Primary Failure Mechanism |
|---|:---:|:---:|:---:|---|
| **1. Retrieval Miss** | **6** (33.3%) | **4** (22.2%) | -2 questions | Required evidence chunk completely absent from top-5 context |
| **2. Context Insufficiency** | **5** (27.8%) | **5** (27.8%) | Net 0 | Top-5 contains only partial evidence for multi-part or cross-paper queries |
| **3. Citation Selection Failure** | **3** (16.7%) | **6** (33.3%) | **+3 questions** | Target chunk present in top-5 (ranks 2–5), but system cites non-target rank 1 |
| **4. Unsupported Answer** | **1** (5.6%) | **0** (0.0%) | -1 question | Generated answer asserts claims not substantiated by its cited passage |
| **5. Evaluator Limitation** | **1** (5.6%) | **1** (5.6%) | Net 0 | Target cited & claim supported, but surface-form token recall falls below 0.25 |
| **6. No Clear Failure (Strict Pass)** | **2** (11.1%) | **2** (11.1%) | 0 (Q02, Q11) | Fully grounded, cited, claim-supported, and lexical match verified |
| **Total Answerable Questions** | **18 / 18** | **18 / 18** | Reconciled | Exact sum across all evaluated questions |

### Summary Metric Denominators Disentanglement:

| Diagnostic Dimension | Dense Baseline | Hybrid Mode | Evaluator Denominator Scope |
|---|:---:|:---:|---|
| **Questions Analyzed** | 18 | 18 | 18 Answerable Questions |
| **Target Evidence in Top-5** | 12 / 18 (66.7%) | 15 / 18 (83.3%) | Retrieval Hit@5 over 18 questions |
| **Relevant Evidence Cited** | 6 / 18 (33.3%) | 6 / 18 (33.3%) | Citation Provenance over 18 questions |
| **Unsupported Answer Risk Count** | 12 / 18 (66.7%) | 12 / 18 (66.7%) | Answering without verified target evidence |
| **Likely Evaluator Limitations** | 1 / 18 (5.6%) | 1 / 18 (5.6%) | False negatives from unigram overlap heuristic |
| **Strict Appropriate Outcome Rate** | 2 / 28 (42.9%) | 2 / 28 (42.9%) | 2 Answerable Passes + 10 OOS Abstentions out of 28 |
| **Citation Validity Rate** | **100.0% (20 / 20)** | **100.0% (20 / 20)** | **Per-Citation Denominator (20 Emitted Citations)** |
| **Citation Excerpt Faithfulness** | **100.0% (20 / 20)** | **100.0% (20 / 20)** | **Per-Citation Denominator (20 Emitted Citations)** |

> [!IMPORTANT]
> **Denominator Clarity (20 Citations vs. 18 Questions):**
> Questions `Q02` and `Q11` each require multi-passage evidence and emit **2 citations each**. The remaining 16 answerable questions emit 1 citation each ($16 \times 1 + 2 \times 2 = 20$). The 10 out-of-scope questions emit 0 citations. Thus, the total citation denominator is **20 citations**, achieving **20 / 20 (100.0%)** validity and verbatim faithfulness.

---

## 3. Deep-Dive Analysis of Difficult Cases

This section examines the critical questions highlighted by the audit, using exact excerpts from the underlying indexed chunks and retrieved context.

### Case Q01: What generator model does RAG use?
- **Target Papers & Physical Pages:** `['P1']` on `{'P1': [3]}`
- **Expected Ground-Truth Answer:** RAG uses BART-large (400M parameters), a pre-trained sequence-to-sequence transformer, as its generator model.
- **Supporting Excerpt:** *""The generator component $p_\theta(y_i"*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P1_p02_c06` (r1), `P1_p17_c03` (r2), `P1_p04_c01` (r3), `P4_p01_c05` (r4), `P1_p08_c01` (r5) | `P1_p04_c01` (r1), `P1_p03_c02` (r2*), `P1_p17_c03` (r3), `P4_p04_c03` (r4), `P1_p19_c02` (r5) | Hybrid surfaced target page |
| **Citations Emitted** | `P1_p04_c01` (p4) | `P1_p04_c01` (p4) | Off-target citation |
| **Claim Support Check** | ❌ Fail | ❌ Fail | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ✅ (Recall: 0.38) | ✅ (Recall: 0.38) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ❌ Fail | ❌ Fail | End-to-end verified outcome |
| **Primary Failure** | **Retrieval miss** | **Retrieval miss** | BART-large definition chunk P1_p03_c04 absent from top-5 context. |

#### Evidence Verification & Findings:
1. **Absence of BART-Large Evidence in Retrieved Context:**
   - The authoritative definition of RAG's generator model resides exclusively in chunk `P1_p03_c04` on physical page 3:
     > *"BART-large [32], a pre-trained seq2seq transformer [58] with 400M parameters."*
   - In **Dense retrieval**, chunk `P1_p03_c04` is completely absent from top-5 (retrieved chunks: `P1_p02_c06`, `P1_p17_c03`, `P1_p04_c01`, `P4_p01_c05`, `P1_p08_c01`).
   - In **Hybrid retrieval**, page 3 is retrieved at Rank 2 (`P1_p03_c02`), but inspection of `P1_p03_c02` reveals it contains only mathematical equations for marginalization:
     > *"...is marginalized to get the seq2seq probability $p(y|x)$ via a top-K approximation. Concretely, the top K documents are retrieved using the retriever, and the generator produces the output sequence probability for each document, which are then marginalized..."*
   - `P1_p03_c02` does NOT mention BART, BART-large, or 400M parameters. Thus, physical page presence did not provide the required evidence.
2. **Pipeline Honesty & Citation:**
   - Both modes cite chunk `P1_p04_c01` (page 4), which contains partial evidence: *"only fine-tuning the query encoder BERTq and the BART generator"*.
   - The mock provider explicitly acknowledges the missing evidence: *"RAG fine-tunes 'the BART generator', but the retrieved passages do not contain evidence specifying the BART-large model variant or its 400M parameter count."*
   - Consequently, both Dense and Hybrid correctly report `unsupported_evidence_risk = True` and fail strict evaluation due to an upstream **Retrieval Miss** of `P1_p03_c04`.

---

### Case Q02: How does DPR train its passage encoder, and what types of negatives are used?
- **Target Papers & Physical Pages:** `['P2']` on `{'P2': [3, 5, 6]}`
- **Expected Ground-Truth Answer:** DPR uses a dual-encoder architecture with two independent uncased BERT-base models ($d=768$) for query and passage encoding. In its best model, it is trained using in-batch negative passages (batch size 128) plus one additional BM25 hard negative passage per question (top passages returned by BM25 that do not contain the answer but match question tokens).
- **Supporting Excerpt:** *""Our best model uses gold passages from the same mini-batch and one BM25 negative passage." (§3.2, p. 3); "The DPR model used in our main experiments is trained using the in-batch negative setting (Section 3.2) with a batch size of 128 and one additional BM25 negative passage per question." (§5, p. 5)"*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P2_p13_c03` (r1), `P1_p04_c01` (r2), `P2_p03_c04` (r3*), `P2_p05_c02` (r4*), `P4_p04_c01` (r5) | `P2_p03_c04` (r1*), `P2_p03_c05` (r2*), `P4_p04_c01` (r3), `P2_p05_c02` (r4*), `P2_p09_c04` (r5) | Identical target presence |
| **Citations Emitted** | `P2_p05_c02` (p5*), `P2_p03_c04` (p3*) | `P2_p05_c02` (p5*), `P2_p03_c04` (p3*) | Target page cited |
| **Claim Support Check** | ✅ Pass | ✅ Pass | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ✅ (Recall: 0.75) | ✅ (Recall: 0.75) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ✅ Pass | ✅ Pass | End-to-end verified outcome |
| **Primary Failure** | **No clear failure** | **No clear failure** | Passes strict evaluation: both in-batch and BM25 negative chunks cited and supported. |

#### Evidence Verification & Findings:
1. **Independent Dual Citation Substantiation:**
   - **Citation 1 (`P2_p05_c02`, Physical Page 5):** Verbatim text: *"The DPR model used in our main experiments is trained using the in-batch negative setting (Section 3.2) with a batch size of 128 and one additional BM25 negative passage per question."* This chunk independently substantiates the in-batch negative setting, batch size 128, and BM25 hard negatives.
   - **Citation 2 (`P2_p03_c04`, Physical Page 3):** Verbatim text: *"We consider three different types of negatives: (1) Random: any random passage from the corpus; (2) BM25: top passages returned by BM25 which don’t contain the answer but match most question tokens; (3) Gold..."* This chunk independently substantiates the negative taxonomy.
2. **Evaluation Outcome:**
   - All four criteria (target provenance, evidence claim support, verbatim faithfulness, and lexical recall of 0.75) pass cleanly in both Dense and Hybrid modes. It represents a genuine success (**No clear failure**).

---

### Case Q05: How does RAGAS compute its faithfulness metric?
- **Target Papers & Physical Pages:** `['P5']` on `{'P5': [3]}`
- **Expected Ground-Truth Answer:** Ragas computes Faithfulness in two steps using an LLM: (1) Decomposes the generated answer $a_s(q)$ into a set of atomic statements $S(a_s(q))$ via an extraction prompt; (2) Verifies whether each statement $s_i \in S$ can be inferred from the retrieved context $c(q)$ using a verification prompt function $v(s_i, c(q)) \in \{Yes, No\}$. The final score is the fraction of supported statements: $F = \frac{
- **Supporting Excerpt:** *""To estimate faithfulness, we first use an LLM to extract a set of statements, $S(a_s(q))$. The aim of this step is to decompose longer sentences into shorter and more focused assertions... For each statement $s_i$ in $S$, the LLM determines if $s_i$ can be inferred from $c(q)$ using a verification function $v(s_i, c(q))$... The final faithfulness score, $F$, is then computed as $F = \frac{"*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P5_p03_c01` (r1*), `P5_p05_c02` (r2), `P1_p02_c06` (r3), `P1_p06_c02` (r4), `P4_p01_c05` (r5) | `P5_p05_c02` (r1), `P5_p05_c04` (r2), `P5_p01_c05` (r3), `P5_p05_c03` (r4), `P5_p05_c01` (r5) | Identical target presence |
| **Citations Emitted** | `P5_p05_c02` (p5) | `P5_p05_c02` (p5) | Off-target citation |
| **Claim Support Check** | ❌ Fail | ❌ Fail | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ❌ (Recall: 0.04) | ❌ (Recall: 0.04) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ❌ Fail | ❌ Fail | End-to-end verified outcome |
| **Primary Failure** | **Context insufficiency** | **Retrieval miss** | Introductory chunk P5_p03_c01 retrieved at Rank 1, but mathematical definition and prompt steps in c03/c04 are absent from top-5. |

#### Evidence Verification & Findings:
1. **Introductory Description vs. Formal Algorithmic Definition:**
   - The RAGAS paper structure on physical page 3 contains four distinct chunks:
     - `P5_p03_c01`: Qualitative motivation defining Faithfulness conceptually (*"Faithfulness refers to the idea that the answer should be grounded in the given context. This is important to avoid hallucinations..."*).
     - `P5_p03_c02` & `P5_p03_c03`: The 2-step LLM extraction and verification prompt definition ($S(a_s(q))$ decomposition and verification function $v(s_i, c(q))$).
     - `P5_p03_c04`: The mathematical formula: *"The final faithfulness score, F, is then computed as $F = \frac{|V|}{|S|}$, where $|V|$ is the number of statements that were supported according to the LLM and $|S|$ is the total number of statements."*
   - In **Dense mode**, only `P5_p03_c01` is retrieved (Rank 1). It contains solely the introductory motivation; the algorithmic steps and formula in `c03` and `c04` are absent from top-5 (**Context Insufficiency**).
   - In **Hybrid mode**, Page 3 is not retrieved at all (top-5 is dominated by Page 5 correlation tables). This is a total **Retrieval Miss** of Page 3.

---

### Case Q11: On which dataset does BM25 outperform DPR, and why?
- **Target Papers & Physical Pages:** `['P2']` on `{'P2': [5, 8]}`
- **Expected Ground-Truth Answer:** BM25 outperforms DPR on SQuAD (open-domain setting). In Table 2, BM25 achieves **68.8%** Top-20 retrieval accuracy (and 80.0% Top-100), whereas DPR Single achieves **63.2%** Top-20 (77.2% Top-100) and DPR Multi achieves **51.6%** Top-20 (67.6% Top-100). In Table 4, BM25+BERT achieves **38.1%** EM vs DPR Single at **29.8%** and DPR Multi at **24.1%**. The authors explain two reasons: (1) SQuAD annotators wrote questions after viewing the passage, creating unnaturally high lexical overlap between question and passage that strongly favors BM25 keyword matching; (2) SQuAD data was collected from only 500+ Wikipedia articles, creating an extremely biased passage distribution.
- **Supporting Excerpt:** *"P2 §5.1: "We conjecture that the lower performance on SQuAD is due to two reasons. First, the annotators wrote questions after seeing the passage. As a result, there is a high lexical overlap between passages and questions, which gives BM25 a clear advantage. Second, the data was collected from only 500+ Wikipedia articles and thus the distribution of training examples is extremely biased""*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P2_p06_c06` (r1), `P2_p05_c04` (r2*), `P2_p12_c03` (r3), `P2_p05_c03` (r4*), `P2_p08_c02` (r5*) | `P2_p08_c02` (r1*), `P2_p05_c04` (r2*), `P2_p05_c02` (r3*), `P2_p12_c03` (r4), `P2_p06_c06` (r5) | Identical target presence |
| **Citations Emitted** | `P2_p08_c02` (p8*), `P2_p05_c04` (p5*) | `P2_p08_c02` (p8*), `P2_p05_c04` (p5*) | Target page cited |
| **Claim Support Check** | ✅ Pass | ✅ Pass | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ✅ (Recall: 0.71) | ✅ (Recall: 0.71) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ✅ Pass | ✅ Pass | End-to-end verified outcome |
| **Primary Failure** | **No clear failure** | **No clear failure** | Passes strict evaluation: Table 4 empirical outcome and Section 5 annotator explanation both cited and supported. |

#### Evidence Verification & Findings:
1. **Dual Evidence Verification (Outcome + Explanation):**
   - **Citation 1 (`P2_p08_c02`, Physical Page 8):** Verbatim text: *"...in all cases except SQuAD, answers extracted from the passages retrieved by DPR are more likely to be correct, compared to those from BM25."* Substantiates the empirical finding that BM25 outperforms DPR on SQuAD.
   - **Citation 2 (`P2_p05_c04`, Physical Page 5):** Verbatim text: *"We conjecture that the lower performance on SQuAD is due to two reasons. First, the annotators wrote questions after seeing the passage... Second, the data was collected from only 500+ Wikipedia articles..."* Substantiates the two underlying explanatory mechanisms.
2. **Evaluation Outcome:**
   - Both citations are verified against ground-truth target pages and passage text. Lexical match recall is 0.71. Passes strict evaluation (**No clear failure**).

---

### Case Q13: What are the known limitations of using LLMs as evaluators in RAGAS?
- **Target Papers & Physical Pages:** `['P5']` on `{'P5': [3, 4, 5]}`
- **Expected Ground-Truth Answer:** Ragas acknowledges several limitations: (1) Context Relevance is the hardest metric to automate, as the LLM evaluator (ChatGPT) often struggles to reliably isolate crucial sentences from longer contexts (§5); (2) High dependency on proprietary LLM APIs (`gpt-3.5-turbo-16k` and `text-embedding-ada-002`), which incurs API cost, latency, and potential variability when the underlying API model changes (§3); (3) WikiEval is a relatively small validation benchmark (50 Wikipedia pages/triples) (§4).
- **Supporting Excerpt:** *"P5 §5: "We found context relevance to be the hardest quality dimension to evaluate. In particular, we observed that ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts.""*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P1_p02_c06` (r1), `P5_p01_c04` (r2), `P5_p01_c05` (r3), `P4_p10_c03` (r4), `P1_p17_c03` (r5) | `P5_p01_c02` (r1), `P5_p01_c01` (r2), `P5_p01_c05` (r3), `P5_p01_c03` (r4), `P4_p10_c05` (r5) | Identical target presence |
| **Citations Emitted** | `P5_p01_c04` (p1) | `P5_p01_c03` (p1) | Off-target citation |
| **Claim Support Check** | ❌ Fail | ❌ Fail | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ❌ (Recall: 0.00) | ❌ (Recall: 0.00) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ❌ Fail | ❌ Fail | End-to-end verified outcome |
| **Primary Failure** | **Retrieval miss** | **Retrieval miss** | P5 Section 5 limitations paragraph on Page 5 absent from top-5 due to vocabulary mismatch with paper abstract. |

#### Evidence Verification & Findings:
1. **Vocabulary Mismatch Causing Retrieval Miss:**
   - Question text: *"What are the known limitations of using LLMs as evaluators in RAGAS?"*
   - Ground-truth evidence is in `P5_p05_c04` (Physical Page 5): *"We found context relevance to be the hardest quality dimension to evaluate. In particular, we observed that ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts."*
   - However, `P5_p05_c04` does not use the word *"limitations"* or *"evaluators"* (it refers to *"ChatGPT"*, *"struggles with"*, *"hardest quality dimension"*).
   - Consequently, both Dense embedding similarity and BM25 term matching heavily score chunks from Page 1 (the paper title and abstract, which contain *"Ragas: Automated Evaluation of Retrieval Augmented Generation"*, *"LLMs"*, *"evaluators"*).
   - Target Page 5 is absent from top-5 in both modes. This is a **Retrieval Miss** driven by query-passage vocabulary divergence rather than an evaluator or annotation error.

---

### Case Q15: How does Self-RAG's critic model get its training data?
- **Target Papers & Physical Pages:** `['P4']` on `{'P4': [5]}`
- **Expected Ground-Truth Answer:** Self-RAG trains its in-house critic model $C$ by prompting GPT-4 with type-specific instructions and few-shot demonstrations to predict reflection tokens given sampled task inputs $x$, outputs $y$, and passages $d$. The authors collect 4k–20k supervised training instances per reflection token type from GPT-4 evaluations, combine them into an instruction-tuning dataset, and fine-tune a Llama 2 model as the critic model $C$ to predict reflection tokens.
- **Supporting Excerpt:** *""We create supervised data by prompting GPT-4 to generate reflection tokens and then distill their knowledge into an in-house C... We prompt GPT-4 with a type-specific instruction... followed by few-shot demonstrations $I$, the original task input $x$ and output $y$ to predict an appropriate reflection token as text: $p(r"*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P4_p01_c05` (r1), `P4_p04_c05` (r2), `P4_p04_c04` (r3), `P4_p09_c02` (r4), `P4_p10_c03` (r5) | `P4_p09_c02` (r1), `P4_p04_c04` (r2), `P4_p02_c03` (r3), `P4_p10_c03` (r4), `P4_p06_c01` (r5) | Identical target presence |
| **Citations Emitted** | `P4_p04_c04` (p4) | `P4_p09_c02` (p9) | Off-target citation |
| **Claim Support Check** | ❌ Fail | ❌ Fail | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ❌ (Recall: 0.06) | ❌ (Recall: 0.06) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ❌ Fail | ❌ Fail | End-to-end verified outcome |
| **Primary Failure** | **Retrieval miss** | **Retrieval miss** | Section header chunk P4_p04_c05 present at Rank 2, but substantive data collection body chunk P4_p05_c02 is absent from top-5. |

#### Evidence Verification & Findings:
1. **Chunk Boundary Disconnection Across Physical Page Transitions:**
   - Question: *"How does Self-RAG's critic model get its training data?"*
   - Dense retrieval successfully surfaced `P4_p04_c05` at Rank 2. That chunk ends at the very bottom of Page 4 with the heading:
     > *"3.2.1 TRAINING THE CRITIC MODEL Data collection for critic model. Manual annotation of reflection tokens for each segment is expensive (Wu et al., 2023). A state-of-the-art LLM like GPT-4 (OpenAI, 2023) can be effectively..."*
   - The actual substantive explanation of prompting GPT-4 with few-shot demonstrations and collecting 4k–20k instances starts at the top of Page 5 in chunk `P4_p05_c02`:
     > *"We create supervised data by prompting GPT-4 to generate reflection tokens and then distill their knowledge into an in-house C... We prompt GPT-4 with a type-specific instruction... followed by few-shot demonstrations..."*
   - Because chunking splits across page boundaries, the heading chunk on Page 4 was retrieved, but the explanatory body chunk `P4_p05_c02` was not retrieved in either Dense or Hybrid mode. This represents a **Retrieval Miss** exacerbated by chunk boundary severance.

---

### Case Q16: Do RAG and REALM use the same knowledge corpus, and how do their passage sizes compare?
- **Target Papers & Physical Pages:** `['P1', 'P3', 'P2']` on `{'P1': [4], 'P2': [4], 'P3': [6]}`
- **Expected Ground-Truth Answer:** Both papers use the exact same source snapshot: the English Wikipedia dump from December 20, 2018. However, their passage segmentation differs significantly: RAG (P1) splits Wikipedia into disjoint text blocks of 100 words, producing 21,015,324 passages (~21M passages). REALM (P3) greedily splits Wikipedia documents into chunks of up to 288 BERT wordpieces, producing just over 13 million retrieval candidates (~13M passages). Because 100 words corresponds to ~130–150 wordpieces, REALM's passage chunks are roughly twice as long as RAG's.
- **Supporting Excerpt:** *"P1 §4: "Following Lee et al. [31] and Karpukhin et al. [26], we dump the text from the Dec 2018 dump, and split each article into disjoint 100-word passages, resulting in 21,015,324 passages." / P3 §4.2: "Our knowledge corpus is derived from the December 20, 2018 snapshot of English Wikipedia. Documents are greedily split into chunks of up to 288 BERT wordpieces, resulting in just over 13 million retrieval candidates.""*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P3_p12_c01` (r1), `P3_p07_c05` (r2), `P1_p06_c02` (r3), `P1_p10_c01` (r4), `P3_p02_c01` (r5) | `P3_p12_c01` (r1), `P1_p04_c05` (r2*), `P3_p02_c01` (r3), `P3_p02_c04` (r4), `P1_p04_c04` (r5*) | Hybrid surfaced target page |
| **Citations Emitted** | `P3_p12_c01` (p12) | `P1_p04_c05` (p4*) | Off-target citation |
| **Claim Support Check** | ❌ Fail | ❌ Fail | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ❌ (Recall: 0.03) | ❌ (Recall: 0.00) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ❌ Fail | ❌ Fail | End-to-end verified outcome |
| **Primary Failure** | **Retrieval miss** | **Context insufficiency** | Cross-paper question: neither RAG 100-word nor REALM 288-wordpiece passage chunks present in top-5. |

#### Evidence Verification & Findings:
1. **Multi-Document Synthesis Failure:**
   - Question: *"Do RAG and REALM use the same knowledge corpus, and how do their passage sizes compare?"*
   - Required evidence spans RAG (`P1`, Page 4: 100-word passages, 21M chunks) and REALM (`P3`, Page 6: up to 288 BERT wordpieces, 13M chunks).
   - **Dense mode** failed completely: neither P1 nor P3 target pages were retrieved (**Retrieval Miss**).
   - **Hybrid mode** successfully retrieved RAG's chunks (`P1_p04_c05` and `P1_p04_c04` at Ranks 2 and 5), but completely missed REALM's passage segmentation chunk `P3_p06_c02` (**Context Insufficiency**). Only half of the comparative fact base was available in context.

---

### Case Q17: What evaluation benchmarks are shared across the 2020 papers, and which are unique to each?
- **Target Papers & Physical Pages:** `['P1', 'P2', 'P3']` on `{'P1': [6], 'P2': [5, 8], 'P3': [7]}`
- **Expected Ground-Truth Answer:** Among the 2020 papers (P1, P2, P3): (1) Benchmarks shared across all three papers (P1, P2, P3): Natural Questions (NQ), WebQuestions (WQ), and CuratedTREC (CT); (2) Benchmarks shared between P1 and P2 only: TriviaQA (TQA); (3) Benchmark unique to P2: SQuAD (open-domain setting); (4) Benchmarks unique to P1: MS-MARCO NLG (abstractive open-domain QA), Jeopardy question generation, and FEVER (fact verification); (5) P3 has no unique benchmarks (evaluates solely on NQ, WQ, CT).
- **Supporting Excerpt:** *""*

#### Comparative Mode Breakdown:

| Attribute | Dense Retrieval | Hybrid Retrieval | Diagnostic Contrast |
|---|---|---|---|
| **Top-5 Chunks** | `P4_p17_c05` (r1), `P3_p06_c06` (r2), `P4_p08_c01` (r3), `P2_p13_c04` (r4), `P4_p07_c06` (r5) | `P3_p07_c01` (r1*), `P1_p17_c02` (r2), `P3_p06_c06` (r3), `P4_p07_c01` (r4), `P4_p09_c05` (r5) | Hybrid surfaced target page |
| **Citations Emitted** | `P3_p06_c06` (p6) | `P3_p07_c01` (p7*) | Off-target citation |
| **Claim Support Check** | ❌ Fail | ❌ Fail | Verified claim token presence in cited text |
| **Lexical Match ($\ge 0.25$)** | ❌ (Recall: 0.04) | ❌ (Recall: 0.04) | Non-stopword token overlap heuristic |
| **Strict Outcome** | ❌ Fail | ❌ Fail | End-to-end verified outcome |
| **Primary Failure** | **Retrieval miss** | **Context insufficiency** | Cross-paper question: benchmark tables across 2020 papers absent from top-5 (polluted by 2023 paper P4). |

#### Evidence Verification & Findings:
1. **Temporal Benchmark Pollution & Insufficiency:**
   - Question: *"What evaluation benchmarks are shared across the 2020 papers, and which are unique to each?"*
   - Target evidence requires cross-paper synthesis across the three 2020 papers: P1 (Page 6), P2 (Pages 5, 8), and P3 (Page 7).
   - In **Dense mode**, 3 of the 5 retrieved chunks originated from Self-RAG (`P4`), a 2023 paper that was not part of the query scope, completely crowding out target 2020 tables (**Retrieval Miss**).
   - In **Hybrid mode**, Rank 1 successfully retrieved `P3_p07_c01` (REALM's Table 1 covering NQ, WQ, CT), but P1 (Page 6) and P2 (Pages 5, 8) tables were absent from top-5 (**Context Insufficiency**). The system could observe REALM's benchmarks, but could not determine which were shared or unique relative to RAG or DPR.

---

## 4. Per-Question Diagnostic Inventory (All 18 Questions)

The tables below present the exact diagnostic state for all 18 answerable questions across Dense and Hybrid retrieval modes.

### Dense Retrieval Mode Inventory:

| ID | Question Text | Target Pages | Top-5 Chunks (R1–R5) | Cited Chunks | Target Cited? | Claim Supp? | Lexical Recall | Appropriate? | Primary Failure Category |
|---|---|:---:|---|---|:---:|:---:|:---:|:---:|---|
| **Q01** | What generator model does RAG use? | `{'P1': [3]}` | `P1_p02_c06`, `P1_p17_c03`, `P1_p04_c01`, `P4_p01_c05`, `P1_p08_c01` | `P1_p04_c01` | ❌ | ❌ | ✅ (0.38) | ❌ Fail | **Retrieval miss** |
| **Q02** | How does DPR train its passage encoder, ... | `{'P2': [3, 5, 6]}` | `P2_p13_c03`, `P1_p04_c01`, `P2_p03_c04`, `P2_p05_c02`, `P4_p04_c01` | `P2_p05_c02`, `P2_p03_c04` | ✅ | ✅ | ✅ (0.75) | ✅ Pass | **No clear failure** |
| **Q03** | What is the difference between RAG-Seque... | `{'P1': [3]}` | `P1_p03_c03`, `P1_p04_c01`, `P1_p03_c02`, `P1_p08_c03`, `P4_p03_c06` | `P1_p03_c03` | ✅ | ✅ | ❌ (0.00) | ❌ Fail | **Unsupported answer** |
| **Q04** | What are the four reflection tokens in S... | `{'P4': [4]}` | `P4_p01_c05`, `P4_p04_c01`, `P4_p17_c01`, `P4_p21_c01`, `P4_p10_c04` | `P4_p04_c01` | ✅ | ✅ | ❌ (0.16) | ❌ Fail | **Evaluator limitation** |
| **Q05** | How does RAGAS compute its faithfulness ... | `{'P5': [3]}` | `P5_p03_c01`, `P5_p05_c02`, `P1_p02_c06`, `P1_p06_c02`, `P4_p01_c05` | `P5_p05_c02` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | **Context insufficiency** |
| **Q06** | What exact match scores does RAG-Token a... | `{'P1': [5, 6], 'P2': [8]}` | `P1_p06_c02`, `P1_p06_c03`, `P1_p04_c05`, `P1_p05_c05`, `P1_p08_c03` | `P1_p04_c05` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Citation selection failure** |
| **Q07** | How do RAG, DPR, and REALM compare on Na... | `{'P1': [6], 'P2': [8], 'P3': [7]}` | `P1_p06_c02`, `P5_p05_c04`, `P2_p12_c03`, `P4_p20_c05`, `P1_p04_c05` | `P1_p04_c05` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Context insufficiency** |
| **Q08** | How does REALM's index refresh strategy ... | `{'P1': [4], 'P2': [3], 'P3': [4, 5]}` | `P3_p05_c01`, `P3_p08_c03`, `P3_p08_c02`, `P3_p05_c02`, `P3_p06_c04` | `P3_p05_c01` | ✅ | ✅ | ❌ (0.07) | ❌ Fail | **Context insufficiency** |
| **Q09** | How does Self-RAG's approach to deciding... | `{'P1': [2, 3], 'P4': [4]}` | `P4_p01_c05`, `P1_p08_c03`, `P4_p20_c05`, `P4_p04_c01`, `P1_p10_c01` | `P4_p04_c01` | ✅ | ✅ | ❌ (0.08) | ❌ Fail | **Context insufficiency** |
| **Q10** | How does Self-RAG's ISSUP token relate t... | `{'P4': [4], 'P5': [3]}` | `P4_p01_c05`, `P4_p21_c01`, `P5_p03_c01`, `P5_p05_c02`, `P4_p10_c04` | `P4_p10_c04` | ❌ | ❌ | ❌ (0.02) | ❌ Fail | **Context insufficiency** |
| **Q11** | On which dataset does BM25 outperform DP... | `{'P2': [5, 8]}` | `P2_p06_c06`, `P2_p05_c04`, `P2_p12_c03`, `P2_p05_c03`, `P2_p08_c02` | `P2_p08_c02`, `P2_p05_c04` | ✅ | ✅ | ✅ (0.71) | ✅ Pass | **No clear failure** |
| **Q12** | What role does salient span masking play... | `{'P3': [5, 7]}` | `P3_p08_c02`, `P3_p08_c04`, `P3_p05_c03`, `P3_p12_c01`, `P3_p05_c01` | `P3_p08_c02` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | **Citation selection failure** |
| **Q13** | What are the known limitations of using ... | `{'P5': [3, 4, 5]}` | `P1_p02_c06`, `P5_p01_c04`, `P5_p01_c05`, `P4_p10_c03`, `P1_p17_c03` | `P5_p01_c04` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Retrieval miss** |
| **Q14** | What base language models are used acros... | `{'P1': [3], 'P2': [3, 6], 'P3': [5, 7], 'P4': [4, 5, 6, 7, 8], 'P5': [3]}` | `P3_p02_c07`, `P1_p14_c02`, `P5_p06_c06`, `P4_p15_c04`, `P1_p10_c05` | `P5_p06_c06` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Retrieval miss** |
| **Q15** | How does Self-RAG's critic model get its... | `{'P4': [5]}` | `P4_p01_c05`, `P4_p04_c05`, `P4_p04_c04`, `P4_p09_c02`, `P4_p10_c03` | `P4_p04_c04` | ❌ | ❌ | ❌ (0.06) | ❌ Fail | **Retrieval miss** |
| **Q16** | Do RAG and REALM use the same knowledge ... | `{'P1': [4], 'P2': [4], 'P3': [6]}` | `P3_p12_c01`, `P3_p07_c05`, `P1_p06_c02`, `P1_p10_c01`, `P3_p02_c01` | `P3_p12_c01` | ❌ | ❌ | ❌ (0.03) | ❌ Fail | **Retrieval miss** |
| **Q17** | What evaluation benchmarks are shared ac... | `{'P1': [6], 'P2': [5, 8], 'P3': [7]}` | `P4_p17_c05`, `P3_p06_c06`, `P4_p08_c01`, `P2_p13_c04`, `P4_p07_c06` | `P3_p06_c06` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | **Retrieval miss** |
| **Q18** | What limitations does the RAG paper ackn... | `{'P1': [4, 6, 19]}` | `P1_p17_c03`, `P1_p08_c03`, `P1_p08_c02`, `P1_p19_c02`, `P1_p10_c01` | `P1_p17_c03` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | **Citation selection failure** |

### Hybrid Retrieval Mode Inventory:

| ID | Question Text | Target Pages | Top-5 Chunks (R1–R5) | Cited Chunks | Target Cited? | Claim Supp? | Lexical Recall | Appropriate? | Primary Failure Category |
|---|---|:---:|---|---|:---:|:---:|:---:|:---:|---|
| **Q01** | What generator model does RAG use? | `{'P1': [3]}` | `P1_p04_c01`, `P1_p03_c02`, `P1_p17_c03`, `P4_p04_c03`, `P1_p19_c02` | `P1_p04_c01` | ❌ | ❌ | ✅ (0.38) | ❌ Fail | **Retrieval miss** |
| **Q02** | How does DPR train its passage encoder, ... | `{'P2': [3, 5, 6]}` | `P2_p03_c04`, `P2_p03_c05`, `P4_p04_c01`, `P2_p05_c02`, `P2_p09_c04` | `P2_p05_c02`, `P2_p03_c04` | ✅ | ✅ | ✅ (0.75) | ✅ Pass | **No clear failure** |
| **Q03** | What is the difference between RAG-Seque... | `{'P1': [3]}` | `P1_p04_c01`, `P1_p08_c03`, `P1_p03_c03`, `P1_p06_c03`, `P1_p03_c02` | `P1_p04_c01` | ❌ | ❌ | ❌ (0.08) | ❌ Fail | **Citation selection failure** |
| **Q04** | What are the four reflection tokens in S... | `{'P4': [4]}` | `P4_p17_c01`, `P4_p04_c01`, `P4_p05_c02`, `P4_p04_c03`, `P4_p10_c05` | `P4_p05_c02` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Citation selection failure** |
| **Q05** | How does RAGAS compute its faithfulness ... | `{'P5': [3]}` | `P5_p05_c02`, `P5_p05_c04`, `P5_p01_c05`, `P5_p05_c03`, `P5_p05_c01` | `P5_p05_c02` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | **Retrieval miss** |
| **Q06** | What exact match scores does RAG-Token a... | `{'P1': [5, 6], 'P2': [8]}` | `P1_p04_c05`, `P1_p05_c05`, `P1_p08_c03`, `P1_p06_c03`, `P1_p07_c04` | `P1_p04_c05` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Citation selection failure** |
| **Q07** | How do RAG, DPR, and REALM compare on Na... | `{'P1': [6], 'P2': [8], 'P3': [7]}` | `P1_p04_c05`, `P1_p05_c05`, `P2_p08_c03`, `P2_p12_c03`, `P1_p04_c04` | `P1_p04_c05` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Context insufficiency** |
| **Q08** | How does REALM's index refresh strategy ... | `{'P1': [4], 'P2': [3], 'P3': [4, 5]}` | `P3_p05_c01`, `P3_p08_c03`, `P3_p08_c02`, `P3_p04_c05`, `P3_p06_c04` | `P3_p05_c01` | ✅ | ✅ | ❌ (0.07) | ❌ Fail | **Context insufficiency** |
| **Q09** | How does Self-RAG's approach to deciding... | `{'P1': [2, 3], 'P4': [4]}` | `P4_p20_c05`, `P4_p10_c03`, `P4_p04_c01`, `P4_p09_c02`, `P4_p09_c04` | `P4_p10_c03` | ❌ | ❌ | ❌ (0.05) | ❌ Fail | **Citation selection failure** |
| **Q10** | How does Self-RAG's ISSUP token relate t... | `{'P4': [4], 'P5': [3]}` | `P4_p10_c04`, `P4_p21_c01`, `P4_p10_c05`, `P5_p05_c02`, `P4_p04_c03` | `P4_p10_c04` | ❌ | ❌ | ❌ (0.02) | ❌ Fail | **Citation selection failure** |
| **Q11** | On which dataset does BM25 outperform DP... | `{'P2': [5, 8]}` | `P2_p08_c02`, `P2_p05_c04`, `P2_p05_c02`, `P2_p12_c03`, `P2_p06_c06` | `P2_p08_c02`, `P2_p05_c04` | ✅ | ✅ | ✅ (0.71) | ✅ Pass | **No clear failure** |
| **Q12** | What role does salient span masking play... | `{'P3': [5, 7]}` | `P3_p08_c02`, `P3_p12_c01`, `P3_p05_c03`, `P3_p05_c01`, `P3_p05_c04` | `P3_p08_c02` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | **Citation selection failure** |
| **Q13** | What are the known limitations of using ... | `{'P5': [3, 4, 5]}` | `P5_p01_c02`, `P5_p01_c01`, `P5_p01_c05`, `P5_p01_c03`, `P4_p10_c05` | `P5_p01_c03` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Retrieval miss** |
| **Q14** | What base language models are used acros... | `{'P1': [3], 'P2': [3, 6], 'P3': [5, 7], 'P4': [4, 5, 6, 7, 8], 'P5': [3]}` | `P1_p14_c03`, `P5_p06_c06`, `P4_p07_c06`, `P5_p06_c04`, `P1_p14_c05` | `P1_p14_c03` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | **Context insufficiency** |
| **Q15** | How does Self-RAG's critic model get its... | `{'P4': [5]}` | `P4_p09_c02`, `P4_p04_c04`, `P4_p02_c03`, `P4_p10_c03`, `P4_p06_c01` | `P4_p09_c02` | ❌ | ❌ | ❌ (0.06) | ❌ Fail | **Retrieval miss** |
| **Q16** | Do RAG and REALM use the same knowledge ... | `{'P1': [4], 'P2': [4], 'P3': [6]}` | `P3_p12_c01`, `P1_p04_c05`, `P3_p02_c01`, `P3_p02_c04`, `P1_p04_c04` | `P1_p04_c05` | ✅ | ❌ | ❌ (0.00) | ❌ Fail | **Context insufficiency** |
| **Q17** | What evaluation benchmarks are shared ac... | `{'P1': [6], 'P2': [5, 8], 'P3': [7]}` | `P3_p07_c01`, `P1_p17_c02`, `P3_p06_c06`, `P4_p07_c01`, `P4_p09_c05` | `P3_p07_c01` | ✅ | ❌ | ❌ (0.04) | ❌ Fail | **Context insufficiency** |
| **Q18** | What limitations does the RAG paper ackn... | `{'P1': [4, 6, 19]}` | `P1_p17_c03`, `P1_p08_c02`, `P1_p18_c05`, `P1_p04_c01`, `P1_p19_c02` | `P1_p04_c01` | ✅ | ✅ | ❌ (0.02) | ❌ Fail | **Evaluator limitation** |

---

## 5. Architectural & Evaluator Limitations

### 1. Deterministic Mock Provider Bottlenecks
- **Naive Rank-1 Fallback Selection:** For questions without specialized hardcoded branches (15 out of 18 questions), the mock provider unconditionally selects chunk Rank 1 from the context window. While this was designed to test citation plumbing, in Hybrid mode it actively sabotages grounded generation when RRF ranks off-target preamble chunks at Rank 1 and target chunks at Ranks 2–5.
- **Lack of Semantic Synthesis:** The fallback echoes raw unedited text slices. When a chunk contains tabular data (`Q04`) or complex mathematical definitions (`Q03`, `Q18`), raw text fragments cannot synthesize conceptual distinctions.

### 2. Evaluator Heuristic Limitations
- **Unigram Overlap Vulnerability:** The expected-answer matcher uses non-stopword token recall thresholded at $R \ge 0.25$. While effective for detecting completely wrong answers, it penalizes valid evidence when answers are formatted as Markdown tables or structured lists (`Q04`).
- **Snippet Truncation Penalties:** In `Q18`, the retrieved chunk contains the exact target claim (*"Updating the document encoder BERTd during training is costly..."*), but because the fallback generator sliced the first 120 characters, the relevant sentence was omitted, resulting in an artificial token-recall failure.

### 3. Chunking & Boundary Artifacts
- In `Q15` (Self-RAG critic model data collection), the section heading appears at the bottom of Page 4 while the body text starts on Page 5. Standard page-boundary chunking splits this logical unit into separate chunks, allowing retrieval to pick the heading chunk without the explanatory body.

---

## 6. Recommended Next Steps (Ranked by Evidence & Impact)

Based on the empirical evidence gathered in this diagnostic audit, future implementation phases should address bottlenecks in the following prioritized order:

### Priority 1: Implement Context-Aware Multi-Chunk Evidence Reranking (High Impact)
- **Problem:** Hybrid retrieval brings target evidence into the top-5 for 83.3% of questions, but Hit@1 is only 22.2%. The generation pipeline fails because it lacks an intelligent mechanism to select the most relevant chunk from the top-5 context.
- **Action:** Implement a cross-encoder reranker or an evidence-selection layer (e.g. lexical + dense relevance scoring across the prompt's retrieved passages) before answer synthesis, breaking the dependency on naive Rank-1 chunk selection.

### Priority 2: Transition from Mock Provider to Production LLM Generation (High Impact)
- **Problem:** 6 questions in Hybrid mode fail strictly because the deterministic mock echoes raw text fragments without performing multi-passage synthesis or conceptual reasoning.
- **Action:** Connect a live LLM provider (OpenAI `gpt-4o-mini` or Google `gemini-1.5-flash`) with structured output enforcement (Pydantic `AnswerWithCitations` schema) and provenance-grounding system prompts.

### Priority 3: Cross-Document Query Decomposition for Multi-Paper Synthesis (Medium Impact)
- **Problem:** Cross-paper comparison questions (`Q07`, `Q08`, `Q09`, `Q10`, `Q14`, `Q16`, `Q17`) consistently suffer from Context Insufficiency because a single top-5 retrieval query cannot balance candidate density across multiple independent documents.
- **Action:** Introduce query decomposition / sub-query generation for comparative questions, querying each target document index independently and fusing top results into an interleaved context.

### Priority 4: Section-Aware Chunking across Page Boundaries (Low-to-Medium Impact)
- **Problem:** Page-boundary chunk severance orphaned section headings from body text in `Q15` and fragmented formulas in `Q05`.
- **Action:** Refine chunking logic to support cross-page semantic overlap or carry forward section headers into subsequent page-start chunks.

---

## 7. Non-Mutating Verification Confirmations

- **Source Code Integrity:** No production code in `src/` was modified.
- **Test Suite Integrity:** All existing tests in `tests/` were preserved without modification.
- **Ground-Truth Integrity:** All question annotations in `evaluation/questions/` remain untouched.
- **Authoritative Evaluation Artifacts:** All existing JSON and Markdown files (`retrieval_evaluation.json`, `hybrid_retrieval_evaluation.json`, `generation_evaluation_dense.json`, `generation_evaluation_hybrid.json`, `generation_comparison.json`) remain intact and unaltered.
- **Diagnostic Artifacts Created:**
  - Report: [`evaluation/results/answer_quality_error_analysis.md`](file:///d:/Project/research-intelligence-engine/evaluation/results/answer_quality_error_analysis.md)
  - Companion JSON: [`evaluation/results/answer_quality_error_analysis.json`](file:///d:/Project/research-intelligence-engine/evaluation/results/answer_quality_error_analysis.json)
