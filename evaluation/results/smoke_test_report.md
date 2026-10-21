# LLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-21T09:13:11Z  
**Question**: What generator model does RAG use?  
**Provider**: `gemini` (`gemini-flash-lite-latest`)  
**Retrieval Mode**: `hybrid`  
**Status**: `answered`  

---

## 1. Generated Answer
> In the RAG model, BART is used as the generator.

---

## 2. Provenance & Citations
- **Total Citations Emitted**: 1
- **All Citations in Retrieved Context**: True
- **Citation Excerpts Verified in Stored Source Chunks**: 1 / 1
- **Citation Excerpts Reconciled from Invalid Model Excerpts**: 0

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
| `P1_p04_c01` | `P1` | 4 | Yes | "only ﬁne-tuning the query encoder BERTq and the BART generator." |

---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `True` (token recall: 0.40)
- **Expected Answer Matched**: `False` (token recall: 0.12)
- **Latency**: `1847.9ms`
- **Tokens**: `Prompt: 2370, Completion: 116, Total: 2486`
- **Confidence Notes**: The evidence also notes that the RAG-Token model acts as a standard, autoregressive seq2seq generator.
