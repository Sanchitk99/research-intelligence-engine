"""Read-only dense retrieval diagnostic for selected benchmark questions."""

import json
import re
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ground_truth import load_ground_truth
from src.retrieval.retriever import DenseRetriever


QUESTION_IDS = ["Q01", "Q03", "Q05", "Q06"]
DIAGNOSTIC_TOP_K = 50
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "dense_retrieval_diagnostic_q01_q03_q05_q06.json"

# Atomic facts are intentionally explicit so the diagnostic does not silently
# substitute generated-answer or evaluator results for retrieval evidence.
REQUIRED_FACTS = {
    "Q01": [
        "BART-large",
        "400M parameters",
        "pre-trained sequence-to-sequence transformer",
    ],
    "Q03": [
        "same retrieved document generates the complete sequence",
        "retrieved document is a single latent variable",
        "RAG-Token uses a per-token document latent variable",
    ],
    "Q05": [
        "LLM extracts atomic statements from the generated answer",
        "each statement is verified against the retrieved context",
        "faithfulness is the fraction of supported statements",
    ],
    "Q06": [
        "RAG-Token achieves 44.1 Exact Match on Natural Questions",
        "RAG-Sequence achieves 44.5 Exact Match on Natural Questions",
        "DPR extractive reader achieves 41.5 Exact Match on Natural Questions",
        "DPR achieves 57.9 and RAG-Token achieves 55.2 on standard TriviaQA",
        "RAG-Sequence achieves 68.0 and RAG-Token achieves 66.1 on TQA-Wiki",
    ],
}


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def fact_tokens(fact: str) -> set[str]:
    return set(normalize(fact).split())


def fact_supported(fact: str, text: str) -> bool:
    source = normalize(text)
    tokens = fact_tokens(fact)
    # Require all discriminating words/numbers. This is a diagnostic signal,
    # not a change to benchmark scoring.
    return tokens.issubset(set(source.split()))


def chunk_record(chunk, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "physical_page_number": chunk.physical_page_number,
        "printed_page_number": chunk.printed_page_number,
        "score": chunk.score,
        "text": chunk.text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.diagnostic_top_k < 6:
        parser.error("--diagnostic-top-k must be greater than the main top-five search")
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    questions = {item["id"]: item for item in json.loads((ROOT / "evaluation/questions/answerable_questions.json").read_text(encoding="utf-8"))}
    targets = {target.question_id: target for target in load_ground_truth(ROOT / "evaluation/questions/answerable_questions.json")}
    retriever = DenseRetriever()
    output = {
        "diagnostic": "dense retrieval only; no generation/provider calls",
        "retrieval_mode": "dense",
        "main_top_k": 5,
        "diagnostic_top_k": args.diagnostic_top_k,
        "question_ids": QUESTION_IDS,
        "questions": [],
    }
    try:
        for qid in QUESTION_IDS:
            item = questions[qid]
            target = targets[qid]
            top5_result = retriever.retrieve(item["question"], top_k=5)
            diagnostic_result = retriever.retrieve(item["question"], top_k=args.diagnostic_top_k)
            top5 = [chunk_record(chunk, rank) for rank, chunk in enumerate(top5_result.chunks, 1)]
            diagnostic = [chunk_record(chunk, rank) for rank, chunk in enumerate(diagnostic_result.chunks, 1)]
            facts = []
            for fact in REQUIRED_FACTS[qid]:
                top5_hits = [entry for entry in top5 if fact_supported(fact, entry["text"])]
                all_hits = [entry for entry in diagnostic if fact_supported(fact, entry["text"])]
                facts.append({
                    "fact": fact,
                    "supported_by_top_five": bool(top5_hits),
                    "top_five_ranks": [entry["rank"] for entry in top5_hits],
                    "diagnostic_rank_if_found": all_hits[0]["rank"] if all_hits else None,
                    "diagnostic_chunk_id_if_found": all_hits[0]["chunk_id"] if all_hits else None,
                })
            top_count = sum(1 for fact in facts if fact["supported_by_top_five"])
            if top_count == len(facts):
                classification = "evidence present in top five"
            elif top_count > 0:
                classification = "evidence partially present"
            elif any(fact["diagnostic_rank_if_found"] is not None for fact in facts):
                classification = "evidence missing from top five"
            else:
                classification = "unable to determine"
            output["questions"].append({
                "id": qid,
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "source_papers": item.get("source_papers", []),
                "page_references": item.get("page_references", ""),
                "supporting_excerpt": item.get("supporting_excerpt", ""),
                "parsed_target_pages": target.expected_pages,
                "required_facts": facts,
                "all_required_facts_supported_by_top_five": all(fact["supported_by_top_five"] for fact in facts),
                "classification": classification,
                "top_five": top5,
                "diagnostic_search_top_50": diagnostic,
                "top_five_execution_time_ms": top5_result.execution_time_ms,
                "diagnostic_execution_time_ms": diagnostic_result.execution_time_ms,
            })
    finally:
        retriever.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    for question in output["questions"]:
        print(f"{question['id']}: {question['classification']}; all_top5={question['all_required_facts_supported_by_top_five']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
