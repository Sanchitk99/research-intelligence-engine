# LLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-20T02:30:01Z  
**Question**: What are the four reflection tokens in Self-RAG and what does each one assess?  
**Provider**: `groq` (`openai/gpt-oss-20b`)  
**Retrieval Mode**: `dense`  
**Status**: `error`  

---

## 1. Generated Answer
> LLM provider error encountered: Groq API returned error HTTP 400: {"error":{"message":"Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.","type":"invalid_request_error","code":"json_validate_failed","failed_generation":""}}


---

## 2. Provenance & Citations
- **Total Citations Emitted**: 0
- **All Citations in Retrieved Context**: True
- **Citation Excerpts Verified in Stored Source Chunks**: 0 / 0
- **Citation Excerpts Reconciled from Invalid Model Excerpts**: 0

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |


---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `False` (token recall: 0.00)
- **Expected Answer Matched**: `False` (token recall: 0.00)
- **Latency**: `3139.9ms`
- **Tokens**: `N/A`
- **Confidence Notes**: Groq API returned error HTTP 400: {"error":{"message":"Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.","type":"invalid_request_error","code":"json_validate_failed","failed_generation":""}}

