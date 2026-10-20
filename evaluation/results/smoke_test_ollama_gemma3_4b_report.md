# LLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-20T03:16:49Z  
**Question**: What generator model does RAG use?  
**Provider**: `ollama` (`gemma3:4b`)  
**Retrieval Mode**: `hybrid`  
**Status**: `answered`  

---

## 1. Generated Answer
> The RAG-Token model uses a standard, autoregressive seq2seq generator with transition probability: p′θ(yi|x, y1:i−1) = Pz∈top-k(p(·|x)) pη(zi|x)pθ(yi|x, zi, y1:i−1). The RAG-Sequence model uses a top-K approximation to marginalize the likelihood. Both models utilize a BART generator.

---

## 2. Provenance & Citations
- **Total Citations Emitted**: 3
- **All Citations in Retrieved Context**: True
- **Citation Excerpts Verified in Stored Source Chunks**: 3 / 3
- **Citation Excerpts Reconciled from Invalid Model Excerpts**: 1

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
| `P1_p03_c02` | `P1` | 3 | Yes | "is marginalized to get the seq2seq probability p(y|x) via a top-K approximation. Concretely, the
top K documents are ret..." |
| `P1_p04_c01` | `P1` | 4 | Yes | "We do not ﬁnd this step necessary for strong performance, and keep the document encoder (and index) ﬁxed, only ﬁne-tuning the query encoder BERTq and the BART generator." |
| `P1_p19_c02` | `P1` | 19 | Yes | "The non-parametric memory index does not consist of trainable parameters, but does consists of 21M 728 dimensional vectors, consisting of 15.3B values." |

---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `True` (token recall: 0.60)
- **Expected Answer Matched**: `True` (token recall: 0.25)
- **Latency**: `11848.9ms`
- **Tokens**: `Prompt: 2385, Completion: 390, Total: 2775`
- **Confidence Notes**: The evidence primarily describes the architecture and training of the RAG-Token and RAG-Sequence models, detailing the use of a BART generator. (Citation excerpt for 'P1_p03_c02' was not found in passage text; using verified passage prefix.)
