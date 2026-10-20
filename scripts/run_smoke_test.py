"""Controlled one-question LLM smoke-test runner for the Research Intelligence Engine.

Executes a single-question integration test against the configured LLM provider
without running the full 28-question evaluation benchmark or overwriting historical artifacts.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.generation.generator import AnswerGenerator
from src.generation.models import AnswerStatus, GenerationConfig
from src.generation.providers.factory import get_llm_provider
from src.generation.providers.base import ProviderError, ProviderAuthError, ProviderTimeoutError, RateLimitError
from src.evaluation.generation_evaluator import (
    check_evidence_claim_support,
    evaluate_lexical_answer_match,
)


def get_redacted_key_status(key_name: str) -> str:
    """Check environment variable and return redacted status string without exposing secrets."""
    val = os.environ.get(key_name)
    if not val:
        return "Not Set"
    return f"Configured ([VALUE HIDDEN], {len(val)} chars)"


def check_ollama_server(base_url: str = "http://localhost:11434") -> Tuple[bool, Optional[str]]:
    """Check if local Ollama server is reachable."""
    import httpx
    clean_url = base_url.rstrip("/")
    # Check tags endpoint
    try:
        r = httpx.get(f"{clean_url}/api/tags", timeout=1.5)
        return r.status_code == 200, None
    except Exception as e:
        return False, str(e)


def run_smoke_test(
    question: str = "What generator model does RAG use?",
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    dry_run: bool = False,
    output_json: Optional[Path] = None,
    output_md: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute controlled smoke test and record metrics."""
    print("=" * 76)
    print(" Research Intelligence Engine -- Controlled LLM Smoke Test (Task 4.4.2)")
    print("=" * 76)

    # 1. Audit Provider Configuration
    target_provider = (provider_type or os.environ.get("LLM_PROVIDER", "auto")).lower()
    openai_key_status = get_redacted_key_status("OPENAI_API_KEY")
    gemini_key_status = get_redacted_key_status("GEMINI_API_KEY")
    groq_key_status = get_redacted_key_status("GROQ_API_KEY")

    print(f"Requested Provider:     {target_provider}")
    print(f"Retrieval Mode:         {retrieval_mode}")
    print(f"OPENAI_API_KEY Status:  {openai_key_status}")
    print(f"GEMINI_API_KEY Status:  {gemini_key_status}")
    print(f"GROQ_API_KEY Status:    {groq_key_status}")

    # Safety check: if live provider requested but unconfigured, abort cleanly without silent mock fallback
    if target_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        err_msg = (
            "ABORTED: Live provider 'openai' was explicitly requested, but OPENAI_API_KEY is not configured.\n"
            "Safety invariant: Will NOT silently fall back to mock provider during a requested live test.\n"
            "Setup step: Set OPENAI_API_KEY in your environment before running live evaluation."
        )
        print(f"\n[ERROR] {err_msg}")
        return {"error": err_msg, "status": "aborted_missing_credentials"}

    if target_provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        err_msg = (
            "ABORTED: Live provider 'gemini' was explicitly requested, but GEMINI_API_KEY is not configured.\n"
            "Safety invariant: Will NOT silently fall back to mock provider during a requested live test.\n"
            "Setup step: Set GEMINI_API_KEY in your environment before running live evaluation."
        )
        print(f"\n[ERROR] {err_msg}")
        return {"error": err_msg, "status": "aborted_missing_credentials"}

    if target_provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        err_msg = (
            "ABORTED: Live provider 'groq' was explicitly requested, but GROQ_API_KEY is not configured.\n"
            "Safety invariant: Will NOT silently fall back to another provider during a requested live test.\n"
            "Setup step: Set GROQ_API_KEY in your environment before running the smoke test."
        )
        print(f"\n[ERROR] {err_msg}")
        return {"error": err_msg, "status": "aborted_missing_credentials"}

    if target_provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        check_url = base_url.replace("/v1", "")
        reachable, reach_err = check_ollama_server(check_url)
        print(f"Ollama Server ({check_url}): {'Reachable' if reachable else f'Unreachable ({reach_err})'}")
        if not reachable:
            err_msg = (
                f"ABORTED: Ollama server at '{check_url}' is not reachable.\n"
                f"Connection error: {reach_err}\n"
                "Setup step: Ensure Ollama is installed ('ollama serve') and model is pulled ('ollama pull llama3')."
            )
            print(f"\n[ERROR] {err_msg}")
            return {"error": err_msg, "status": "aborted_ollama_unreachable"}

    # 2. Instantiate Generator
    try:
        provider = get_llm_provider(provider_type=provider_type, model_name=model_name)
    except Exception as e:
        print(f"\n[ERROR] Provider initialization failed: {e}")
        return {"error": str(e), "status": "provider_init_failed"}

    print(f"Active Provider:        {provider.name}")
    print(f"Active Model:           {provider.default_model}")
    print(f"Question:               {question}")
    print("-" * 76)

    generator = AnswerGenerator(
        provider=provider,
        retrieval_mode=retrieval_mode,
        config=GenerationConfig(top_k=top_k, model_name=provider.default_model),
    )

    # 3. Execute Retrieval
    retrieval_start = time.perf_counter()
    retrieval_result = generator.retriever.retrieve(query=question, top_k=top_k)
    retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000.0

    print(f"\nRetrieved {len(retrieval_result.chunks)} context chunks ({retrieval_time_ms:.1f}ms):")
    chunks_by_id = {c.chunk_id: c for c in retrieval_result.chunks}
    for i, c in enumerate(retrieval_result.chunks, 1):
        print(f"  [{i}] {c.chunk_id} | Page {c.physical_page_number} | Score: {c.score:.4f} | Preview: {c.text[:80].strip()}...")

    if dry_run:
        print("\n[DRY RUN COMPLETE] Retrieval validated. Stopping before LLM generation.")
        generator.close()
        return {
            "status": "dry_run_success",
            "provider_name": provider.name,
            "model_name": provider.default_model,
            "retrieval_mode": retrieval_mode,
            "retrieved_chunks": [c.chunk_id for c in retrieval_result.chunks],
        }

    # 4. Execute Generation
    print(f"\nInvoking AnswerGenerator ({provider.name} / {provider.default_model})...")
    gen_answer = generator.generate(question=question, retrieval_result=retrieval_result)
    generator.close()

    # 5. Evaluate and Verify Grounding & Provenance
    # Target answer for Q01 from benchmark
    q01_expected = "RAG uses BART-large (400M parameters), a pre-trained sequence-to-sequence transformer, as its generator model."
    q01_excerpt = "BART-large model with 400M parameters"

    # Citation validation against context
    citations_analysis = []
    all_citations_in_context = True
    grounded_excerpt_count = 0
    reconciled_excerpt_count = 0
    for c in gen_answer.citations:
        in_context = c.chunk_id in chunks_by_id
        if not in_context:
            all_citations_in_context = False
        excerpt_text = " ".join(c.excerpt.strip(".\"'\n").split()).lower()
        source_text = " ".join(chunks_by_id[c.chunk_id].text.split()).lower() if in_context else ""
        excerpt_verified = bool(in_context and (excerpt_text in source_text or len(excerpt_text) < 15))
        if excerpt_verified:
            grounded_excerpt_count += 1
        if gen_answer.confidence_notes and f"Citation excerpt for '{c.chunk_id}' was not found" in gen_answer.confidence_notes:
            reconciled_excerpt_count += 1
        citations_analysis.append({
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "physical_page_number": c.physical_page_number,
            "excerpt": c.excerpt,
            "formatted": c.formatted_citation(),
            "in_retrieved_context": in_context,
            "excerpt_verified_in_chunk": excerpt_verified,
        })

    # Evidence claim support
    cited_text = " ".join(chunks_by_id[c.chunk_id].text for c in gen_answer.citations if c.chunk_id in chunks_by_id)
    claim_supported, claim_recall = check_evidence_claim_support(
        cited_chunks_text=cited_text,
        expected_answer=q01_expected,
        question_text=question,
        supporting_excerpt=q01_excerpt,
    )

    # Lexical match
    ans_matched, ans_recall, matched_tokens = evaluate_lexical_answer_match(
        generated_text=gen_answer.answer,
        expected_text=q01_expected,
        question_text=question,
    )

    smoke_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": question,
        "provider_name": gen_answer.provider_name,
        "model_name": gen_answer.model_name,
        "retrieval_mode": gen_answer.retrieval_mode,
        "answer_status": gen_answer.status.value,
        "answer_text": gen_answer.answer,
        "confidence_notes": gen_answer.confidence_notes,
        "latency_ms": gen_answer.latency_ms,
        "prompt_tokens": gen_answer.prompt_tokens,
        "completion_tokens": gen_answer.completion_tokens,
        "total_tokens": gen_answer.total_tokens,
        "citations_count": len(gen_answer.citations),
        "all_citations_in_context": all_citations_in_context,
        "citation_excerpt_verified_count": grounded_excerpt_count,
        "citation_excerpts_all_verified": grounded_excerpt_count == len(gen_answer.citations),
        "citation_excerpts_reconciled_count": reconciled_excerpt_count,
        "citations": citations_analysis,
        "evidence_claim_supported": claim_supported,
        "claim_token_recall": claim_recall,
        "expected_answer_matched": ans_matched,
        "answer_token_recall": ans_recall,
        "retrieved_chunk_ids": [c.chunk_id for c in retrieval_result.chunks],
    }

    # 6. Print Report Table
    print("\n" + "=" * 76)
    print(" SMOKE TEST RESULT")
    print("=" * 76)
    print(f"Provider:                {gen_answer.provider_name} (Model: {gen_answer.model_name})")
    print(f"Status:                  {gen_answer.status.value}")
    print(f"Answer:                  {gen_answer.answer}")
    print(f"Latency:                 {gen_answer.latency_ms:.1f}ms")
    tokens_str = f"Prompt: {gen_answer.prompt_tokens}, Completion: {gen_answer.completion_tokens}, Total: {gen_answer.total_tokens}" if gen_answer.total_tokens else "N/A"
    print(f"Token Usage:             {tokens_str}")
    print(f"Citations ({len(gen_answer.citations)}):")
    for cit in citations_analysis:
        ctx_flag = "VALID (in context)" if cit["in_retrieved_context"] else "INVALID (hallucinated)"
        print(f"  - {cit['formatted']} [{ctx_flag}]")
        print(f"    Excerpt: \"{cit['excerpt']}\"")
    print(f"Evidence Claim Supported: {claim_supported} (Recall: {claim_recall:.2f})")
    print(f"Expected Answer Match:   {ans_matched} (Recall: {ans_recall:.2f})")
    if gen_answer.confidence_notes:
        print(f"Confidence Notes:        {gen_answer.confidence_notes}")
    print("=" * 76)

    # 7. Save outputs
    out_json = output_json or Path("evaluation/results/smoke_test_result.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(smoke_record, f, indent=2)
    print(f"Smoke test JSON saved:   {out_json}")

    out_md = output_md or Path("evaluation/results/smoke_test_report.md")
    md_content = f"""# LLM Integration Smoke Test Report

**Execution Timestamp**: {smoke_record['timestamp']}  
**Question**: {question}  
**Provider**: `{gen_answer.provider_name}` (`{gen_answer.model_name}`)  
**Retrieval Mode**: `{gen_answer.retrieval_mode}`  
**Status**: `{gen_answer.status.value}`  

---

## 1. Generated Answer
> {gen_answer.answer}

---

## 2. Provenance & Citations
- **Total Citations Emitted**: {len(gen_answer.citations)}
- **All Citations in Retrieved Context**: {all_citations_in_context}
- **Citation Excerpts Verified in Stored Source Chunks**: {grounded_excerpt_count} / {len(gen_answer.citations)}
- **Citation Excerpts Reconciled from Invalid Model Excerpts**: {reconciled_excerpt_count}

| Chunk ID | Document | Physical Page | In Context | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(
        f"| `{c['chunk_id']}` | `{c['document_id']}` | {c['physical_page_number']} | {'Yes' if c['in_retrieved_context'] else 'No'} | \"{c['excerpt']}\" |"
        for c in citations_analysis
    ) + f"""

---

## 3. Evaluation & Validation
- **Evidence Claim Supported**: `{claim_supported}` (token recall: {claim_recall:.2f})
- **Expected Answer Matched**: `{ans_matched}` (token recall: {ans_recall:.2f})
- **Latency**: `{gen_answer.latency_ms:.1f}ms`
- **Tokens**: `{tokens_str}`
- **Confidence Notes**: {gen_answer.confidence_notes or "None"}
"""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Smoke test Report saved: {out_md}")
    print("=" * 76)

    return smoke_record


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Controlled LLM Smoke Test Runner")
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openai", "gemini", "groq", "ollama"],
        default="auto",
        help="Provider to test (default: auto)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K context chunks")
    parser.add_argument(
        "--question",
        type=str,
        default="What generator model does RAG use?",
        help="Test question",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and retrieval without calling LLM",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON file path (default: evaluation/results/smoke_test_result.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown file path (default: evaluation/results/smoke_test_report.md)",
    )

    args = parser.parse_args()
    res = run_smoke_test(
        question=args.question,
        provider_type=args.provider if args.provider != "auto" else None,
        model_name=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        dry_run=args.dry_run,
        output_json=args.output_json,
        output_md=args.output_md,
    )

    if res.get("status") in ("aborted_missing_credentials", "aborted_ollama_unreachable", "provider_init_failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
