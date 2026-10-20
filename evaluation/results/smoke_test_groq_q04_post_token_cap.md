# LLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-20T02:28:08Z  
**Question**: What is the difference between RAG-Sequence and RAG-Token?  
**Provider**: `groq` (`openai/gpt-oss-20b`)  
**Retrieval Mode**: `dense`  
**Status**: `answered`  

---

## 1. Generated Answer
> RAG‑Token marginalizes over documents for each output token, letting the generator pick from a different document for every token. This per‑token marginalization lets RAG‑Token be decoded with a standard beam search. In contrast, RAG‑Sequence marginalizes over documents for the entire output sequence; its likelihood cannot be decomposed into a per‑token form, so it cannot be solved with a single beam search and requires a different decoding strategy. Empirically, RAG‑Sequence continues to improve as more documents are retrieved, whereas RAG‑Token’s performance peaks around ten retrieved documents and then degrades.

---

## 2. Provenance & Citations
- **Total Citations Emitted**: 3
- **All Citations in Retrieved Context**: True
- **Citation Excerpts Verified in Stored Source Chunks**: 3 / 3
- **Citation Excerpts Reconciled from Invalid Model Excerpts**: 1

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
| `P1_p03_c02` | `P1` | 3 | Yes | "In the RAG-Token model we can draw a different latent document for each target token and marginalize accordingly. This allows the generator to choose content from several documents when producing an answer." |
| `P1_p04_c01` | `P1` | 4 | Yes | "minimize the negative marginal log-likelihood of each target, P
j −log p(yj|xj) using stochastic
gradient descent with A..." |
| `P1_p08_c03` | `P1` | 8 | Yes | "Figure 3 (left) shows that retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence, but performance peaks for RAG-Token at 10 retrieved documents." |

---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `True` (token recall: 0.40)
- **Expected Answer Matched**: `False` (token recall: 0.10)
- **Latency**: `2245.1ms`
- **Tokens**: `Prompt: 2633, Completion: 1317, Total: 3950`
- **Confidence Notes**: The answer is based solely on the provided passages; no external information was used. (Citation excerpt for 'P1_p04_c01' was not found in passage text; using verified passage prefix.)
