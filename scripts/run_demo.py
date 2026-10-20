"""End-to-End Demonstrable Workflow CLI for the Research Intelligence Engine.

Demonstrates:
  Step 1: Paper Ingestion & Page-Aware Chunking (PDF -> pages -> chunks)
  Step 2: Vector Indexing into Qdrant & BM25 synchronization
  Step 3: Research Question Answering with Hybrid/Dense Retrieval
  Step 4: Citation Provenance Verification (Document, Physical Page, Excerpt)
  Step 5: Safe Abstention on Out-of-Scope / Insufficient Evidence Questions

Usage:
  python scripts/run_demo.py --provider auto
  python scripts/run_demo.py --provider gemini
  python scripts/run_demo.py --provider mock
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe unicode encoding configuration for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.ui.service import ResearchEngineService, IngestionReport
from src.generation.models import AnswerStatus, GeneratedAnswer


def run_demo(provider_name: str = "auto", retrieval_mode: str = "hybrid", top_k: int = 5) -> int:
    """Execute complete end-to-end demonstration workflow."""
    print("=" * 76)
    print(" Research Intelligence Engine -- End-to-End Workflow Demonstration")
    print("=" * 76)
    print(f"Requested Provider: {provider_name}")
    print(f"Retrieval Mode:     {retrieval_mode}")
    print(f"Top-K Chunks:       {top_k}")
    print("-" * 76)

    # Initialize Service
    service = ResearchEngineService(
        qdrant_path=PROJECT_ROOT / "data" / "qdrant",
        collection_name="research_corpus",
        chunks_file=PROJECT_ROOT / "data" / "chunked" / "corpus_chunks.json",
    )

    try:
        # -------------------------------------------------------------------
        # Step 1 & 2: Ingestion & Indexing Demonstration
        # -------------------------------------------------------------------
        print("\n[STEP 1 & 2] Demonstrating Paper Ingestion & Indexing Pipeline")
        sample_paper = PROJECT_ROOT / "evaluation" / "corpus" / "papers" / "P5_RAGAS_Es_2023.pdf"
        print(f"  Target Document: {sample_paper.name}")

        ingest_start = time.perf_counter()
        # Ingest with dry_run=False to test actual extraction, chunking, and point sync
        report: IngestionReport = service.ingest_paper(
            pdf_path=sample_paper,
            doc_id="P5",
            doc_title="Ragas: Automated Evaluation of Retrieval Augmented Generation",
            dry_run=False,
        )
        ingest_dur = (time.perf_counter() - ingest_start) * 1000.0

        if not report.success:
            print(f"  [FAILED] Ingestion failed: {report.error_message}", file=sys.stderr)
            return 1

        print(f"  [SUCCESS] Ingestion & Indexing completed in {ingest_dur:.1f}ms:")
        print(f"    - Document ID:            {report.document_id}")
        print(f"    - Title:                  {report.title}")
        print(f"    - Physical Pages Read:    {report.pages_extracted}")
        print(f"    - Document Chunks Made:   {report.chunks_created}")
        print(f"    - Total Points in Qdrant: {report.total_collection_points}")
        if report.sample_chunks:
            first_c = report.sample_chunks[0]
            print(f"    - Sample Chunk ID:        {first_c['chunk_id']} (Page {first_c['page']})")
            print(f"      Preview: \"{first_c['preview']}\"")

        # -------------------------------------------------------------------
        # Step 3 & 4: Grounded Q&A with Verified Citations
        # -------------------------------------------------------------------
        q1 = "How do RAG-Sequence and RAG-Token differ in generating sequences?"
        print(f"\n[STEP 3 & 4] Grounded Question Answering & Citation Verification")
        print(f"  Question: \"{q1}\"")

        q1_start = time.perf_counter()
        ans1: GeneratedAnswer = service.ask_question(
            question=q1,
            provider_name=provider_name,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
        )
        q1_dur = (time.perf_counter() - q1_start) * 1000.0

        print(f"  Answer Status:  [{ans1.status.value.upper()}]")
        print(f"  Provider/Model: {ans1.provider_name} / {ans1.model_name}")
        print(f"  Latency:        {ans1.latency_ms:.1f}ms (Total roundtrip: {q1_dur:.1f}ms)")
        print(f"  Answer Text:\n    {ans1.answer.strip()}")

        print(f"  Verified Citations ({len(ans1.citations)}):")
        for i, cite in enumerate(ans1.citations, 1):
            printed_info = f", Printed p. {cite.printed_page_number}" if cite.printed_page_number else ""
            print(f"    [{i}] {cite.document_title} ({cite.document_id})")
            print(f"        Physical Page: {cite.physical_page_number}{printed_info} | Chunk: {cite.chunk_id}")
            print(f"        Excerpt: \"{cite.excerpt}\"")

        if ans1.status not in (AnswerStatus.ANSWERED, AnswerStatus.PARTIALLY_ANSWERED):
            print(f"  [WARNING] Expected status ANSWERED/PARTIALLY_ANSWERED, got {ans1.status}")

        # -------------------------------------------------------------------
        # Step 5: Abstention Safety Demonstration
        # -------------------------------------------------------------------
        q2 = "Who won the 2024 ICC Men's T20 World Cup?"
        print(f"\n[STEP 5] Abstention Safety on Out-of-Scope Question")
        print(f"  Question: \"{q2}\"")

        ans2: GeneratedAnswer = service.ask_question(
            question=q2,
            provider_name=provider_name,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
        )

        print(f"  Answer Status:  [{ans2.status.value.upper()}]")
        print(f"  Answer Text:\n    {ans2.answer.strip()}")
        print(f"  Citations Count: {len(ans2.citations)} (Must be 0 on abstention)")

        if len(ans2.citations) > 0:
            print("  [ERROR] Citations must not be attached to an out-of-scope query!", file=sys.stderr)
            return 1

        print("\n" + "=" * 76)
        print(" [SUCCESS] Complete End-to-End Workflow Demonstrated Successfully!")
        print("=" * 76)
        return 0

    finally:
        service.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run End-to-End Workflow Demonstration")
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "gemini", "mock", "openai", "groq", "ollama"],
        help="LLM provider backend (default: auto)",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense"],
        help="Retrieval algorithm (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k context passages (default: 5)",
    )
    args = parser.parse_args()
    return run_demo(
        provider_name=args.provider,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    sys.exit(main())

