# LLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-20T01:18:14Z  
**Question**: What generator model does RAG use?  
**Provider**: `gemini` (`gemini-flash-lite-latest`)  
**Retrieval Mode**: `hybrid`  
**Status**: `answered`  

---

## 1. Generated Answer
> In the RAG model, the BART generator is used (specifically, they keep the document encoder fixed and fine-tune the query encoder and the BART generator).

---

## 2. Provenance & Citations
- **Total Citations Emitted**: 1
- **All Citations in Retrieved Context**: True

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
| `P1_p04_c01` | `P1` | 4 | Yes | "only ﬁne-tuning the query encoder BERTq and the BART generator." |

---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `True` (token recall: 0.40)
- **Expected Answer Matched**: `False` (token recall: 0.12)
- **Latency**: `2387.8ms`
- **Tokens**: `Prompt: 2370, Completion: 113, Total: 2483`
- **Confidence Notes**: None
