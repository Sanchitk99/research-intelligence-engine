# Generation Evaluation Baseline Report (Hybrid Mode)

**Execution Timestamp:** 2026-10-20T05:37:35.030224
**Retrieval Mode:** `hybrid` | **Provider:** `mock` | **Model:** `mock-model` | **Mock Simulation:** `Yes`
**Total Questions Evaluated:** 28 (18 Answerable, 10 Out-of-Scope)

> [!NOTE]
> **Provider & Evaluation Semantics Disclosure:**
> This benchmark was executed using the `mock` provider (`mock-model`) with `hybrid` retrieval.
> Under mock simulation, the system prepends retrieved snippet text without semantic reasoning.
> Answer completion, citation validity, target-page provenance, evidence claim support, and expected-answer matching are strictly separated.

---

## 1. Aggregate Generation Metrics

| Metric | Score | Exact Count | Description |
|---|---|---|---|
| **Out-of-Scope Abstention Accuracy** | **100.0%** | 10 / 10 | Evaluated strictly on out-of-scope questions; correctly abstained |
| **Answerable Completion Rate** | **44.4%** | 8 / 18 | Answered or partially answered (replaces misleading 'grounded rate') |
| **Target Provenance Citation Rate** | **44.4%** | 8 / 18 | Citations include at least one verified ground-truth target paper & page |
| **Evidence Claim Supported Rate** | **44.4%** | 8 / 18 | Cited passages contain the key claims/terms required by ground truth |
| **Expected Answer Match Rate (Lexical)** | **44.4%** | 8 / 18 | Deterministic token-recall match against annotated expected answer |
| **Appropriate Outcome Rate (Strict)** | **64.3%** | 18 / 28 | Strict appropriate outcome: abstention for OOS, verified grounded answer for answerable |
| **Unsupported Answer Risk Count** | **0 / 18** | 0 / 18 | Answerable questions answered despite missing target evidence |
| **Citation Validity Rate** | **100.0%** | 11 / 11 | Emitted citations matching genuine retrieved chunk IDs |
| **Cold-Start Latency** | **23221.4ms** | - | Initial model/index loading roundtrip (isolated_subprocess) |
| **Warm Mean Generation Latency** | **19.4ms** | - | Average steady-state latency across benchmark queries |
| **Warm Median Generation Latency** | **19.1ms** | - | Median steady-state latency across benchmark queries |

---

## 2. Per-Question Evaluation Diagnostics

| ID | Category | Expected | Status | Target Cited? | Evidence Supports? | Lexical Match? | Appropriate? | Valid Cites |
|---|---|---|---|:---:|:---:|:---:|:---:|---|
| **Q01** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q02** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.75) | ✅ Pass | 2 / 2 |
| **Q03** | answerable | answer | `answered` | ✅ | ✅ | ✅ (1.00) | ✅ Pass | 1 / 1 |
| **Q04** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.77) | ✅ Pass | 1 / 1 |
| **Q05** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q06** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q07** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q08** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q09** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.39) | ✅ Pass | 1 / 1 |
| **Q10** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.29) | ✅ Pass | 2 / 2 |
| **Q11** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.71) | ✅ Pass | 2 / 2 |
| **Q12** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.42) | ✅ Pass | 1 / 1 |
| **Q13** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q14** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.02) | ❌ Fail | 0 / 0 |
| **Q15** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q16** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.00) | ❌ Fail | 0 / 0 |
| **Q17** | answerable | answer | `abstained_insufficient_evidence` | ❌ | ❌ | ❌ (0.04) | ❌ Fail | 0 / 0 |
| **Q18** | answerable | answer | `answered` | ✅ | ✅ | ✅ (0.31) | ✅ Pass | 1 / 1 |
| **OOS01** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS02** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS03** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS04** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS05** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS06** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS07** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS08** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS09** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |
| **OOS10** | out_of_scope | abstain | `abstained_out_of_scope` | ❌ | ❌ | ❌ (0.00) | ✅ Pass | 0 / 0 |

---

## 3. Disentanglement & Evaluation Methodology

### Provenance vs. Evidence Grounding vs. Lexical Matching
- **Retrieval / Citation Provenance:** Checks whether citations refer to the physical paper and page containing the ground truth.
- **Evidence Claim Support:** Verifies whether the cited passage text contains key technical terms and claims required to substantiate the answer.
- **Deterministic Expected-Answer Token Match:** Computes non-stopword token recall $R = |tokens(discriminating) \cap tokens(generated)| / |tokens(discriminating)|$ with a deterministic threshold ($R \ge 0.25$) after question token subtraction.

### Latency Measurement Methodology
- **Cold-Start Latency (23221.4ms):** Evaluated via the initial warmup query on an uninitialized pipeline. Measures total duration to load dynamic libraries (PyTorch C++ backend), deserialize SentenceTransformer model weights, open persistent vector storage, build BM25 index (for hybrid), and execute first embedding search. Mode: `isolated_subprocess`.
- **Warm Steady-State Latency (Mean: 19.4ms, Median: 19.1ms):** Computed across the 28 benchmark queries executed strictly after the warmup query, excluding one-time cold-start initialization artifacts.

### Limitations of Deterministic Matching
Deterministic token overlap is a surface-form alignment heuristic, not a proof of semantic entailment:
1. **Negation Blindness:** 'Does not use BART-large' matches tokens identically to 'Uses BART-large'.
2. **Synonym & Paraphrase Penalties:** Legitimate conceptual answers using alternative phrasing may fail token recall.
3. **Claim Independence:** A passage containing keywords in an unrelated section does not prove evidence entailment.

### Metric Denominators: Per-Citation Validity vs. Per-Question Success
- **Per-Citation Denominator (Total Emitted Citations):**
  - **Citation Validity Rate:** Calculated as `total_valid_citations / total_citations_emitted`. Checks whether every citation emitted points to an actual chunk ID present in the retrieved context window.
  - **Citation Excerpt Faithfulness Rate:** Calculated as `total_faithful_excerpts / total_citations_emitted`. Checks whether each cited excerpt is present verbatim within the text of its cited chunk.
- **Per-Question Denominators (Questions Evaluated):**
  - **Answerable Metrics (Denominator: 18 Answerable Questions):** Completion Rate, Target Provenance Citation Rate, Evidence Claim Supported Rate, and Expected Answer Match Rate are computed out of 18 answerable questions.
  - **Out-of-Scope Metrics (Denominator: 10 Out-of-Scope Questions):** Abstention accuracy is computed strictly out of the 10 out-of-scope questions.
  - **Appropriate Outcome Rate (Denominator: 28 Total Evaluated Questions):** Evaluated over all 28 benchmark questions (18 answerable + 10 out-of-scope).

### Out-of-Scope Evaluation
Out-of-scope abstention accuracy is evaluated strictly on the 10 out-of-scope questions (`OOS01` to `OOS10`).
