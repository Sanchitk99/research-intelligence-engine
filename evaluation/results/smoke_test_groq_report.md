# LLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-20T01:47:47Z  
**Question**: What generator model does RAG use?  
**Provider**: `groq` (`openai/gpt-oss-20b`)  
**Retrieval Mode**: `hybrid`  
**Status**: `answered`  

---

## 1. Generated Answer
> RAG uses the BART sequence‑to‑sequence model as its generator.

---

## 2. Provenance & Citations
- **Total Citations Emitted**: 1
- **All Citations in Retrieved Context**: True
- **Citation Excerpts Verified in Stored Source Chunks**: 1 / 1
- **Citation Excerpts Reconciled from Invalid Model Excerpts**: 1

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
| `P1_p04_c01` | `P1` | 4 | Yes | "minimize the negative marginal log-likelihood of each target, P
j −log p(yj|xj) using stochastic
gradient descent with A..." |

---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `True` (token recall: 0.40)
- **Expected Answer Matched**: `True` (token recall: 0.25)
- **Latency**: `1784.1ms`
- **Tokens**: `Prompt: 2313, Completion: 318, Total: 2631`
- **Confidence Notes**: The provided evidence explicitly states that BART is the generator used in RAG. No other generator models are mentioned in the excerpts. (Citation excerpt for 'P1_p04_c01' was not found in passage text; using verified passage prefix.)
