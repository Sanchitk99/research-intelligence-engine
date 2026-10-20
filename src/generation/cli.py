"""Command-line interface for the Answer Generation module."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.generation.models import GenerationConfig, AnswerStatus
from src.generation.generator import AnswerGenerator
from src.generation.providers.factory import get_llm_provider
from src.retrieval.retriever import DenseRetriever


def format_cli_output(ans) -> str:
    """Format GeneratedAnswer for clean terminal presentation."""
    status_icons = {
        AnswerStatus.ANSWERED: "[ANSWERED]",
        AnswerStatus.PARTIALLY_ANSWERED: "[PARTIALLY ANSWERED]",
        AnswerStatus.ABSTAINED_OUT_OF_SCOPE: "[ABSTAINED: OUT OF SCOPE]",
        AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE: "[ABSTAINED: INSUFFICIENT EVIDENCE]",
        AnswerStatus.CONFLICTING_EVIDENCE: "[CONFLICTING EVIDENCE DETECTED]",
        AnswerStatus.ERROR: "[GENERATION ERROR]",
    }

    badge = status_icons.get(ans.status, f"[{ans.status.value.upper()}]")

    lines = []
    lines.append("=" * 72)
    lines.append(" Research Intelligence Engine -- Question Answering")
    lines.append("=" * 72)
    lines.append(f"Question: {ans.question}")
    lines.append(f"Status:   {badge}")
    lines.append(f"Provider: {ans.provider_name} ({ans.model_name}) | Latency: {ans.latency_ms:.1f}ms")
    if ans.total_tokens:
        lines.append(f"Tokens:   Prompt: {ans.prompt_tokens} | Completion: {ans.completion_tokens} | Total: {ans.total_tokens}")
    lines.append("-" * 72)
    lines.append("ANSWER:")
    lines.append(ans.answer)
    lines.append("-" * 72)

    if ans.confidence_notes:
        lines.append("CONFIDENCE & EVIDENCE NOTES:")
        lines.append(f"  {ans.confidence_notes}")
        lines.append("-" * 72)

    lines.append(f"VERIFIED CITATIONS ({len(ans.citations)}):")
    if not ans.citations:
        lines.append("  (No grounded citations)")
    else:
        for i, c in enumerate(ans.citations, 1):
            printed_info = f", Printed Page {c.printed_page_number}" if c.printed_page_number else ""
            lines.append(f"  [{i}] {c.document_id} (Physical Page {c.physical_page_number}{printed_info}) -- Chunk {c.chunk_id}")
            lines.append(f"      Title:   \"{c.document_title}\"")
            lines.append(f"      Excerpt: \"{c.excerpt}\"")

    lines.append("=" * 72)
    return "\n".join(lines)


def main(args: Optional[list] = None) -> int:
    """CLI entrypoint."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Research Intelligence Engine -- Evidence-Grounded Answer Generation"
    )
    parser.add_argument(
        "question",
        type=str,
        help="Research question to answer from the corpus",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "gemini", "groq", "mock", "ollama"],
        help="LLM provider backend (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier override (e.g. gpt-4o-mini, gemini-flash-lite-latest, openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for context (default: 5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum similarity threshold",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response",
    )

    parsed = parser.parse_args(args)

    provider = get_llm_provider(
        provider_type=parsed.provider,
        model_name=parsed.model,
    )

    config = GenerationConfig(
        provider_type=parsed.provider,
        model_name=provider.default_model,
        top_k=parsed.top_k,
        min_score=parsed.min_score,
    )

    generator = AnswerGenerator(
        provider=provider,
        config=config,
    )

    result = generator.generate(
        question=parsed.question,
        top_k=config.top_k,
        min_score=config.min_score,
    )

    if parsed.json:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(format_cli_output(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())

