"""Read-only dense versus hybrid retrieval comparison for selected questions."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


QUESTION_IDS = ["Q01", "Q03", "Q05", "Q06"]
OUTPUT = ROOT / "evaluation" / "results" / "dense_hybrid_retrieval_comparison_q01_q03_q05_q06.json"
DENSE_DIAGNOSTIC = ROOT / "evaluation" / "results" / "dense_retrieval_diagnostic_q01_q03_q05_q06_top100.json"

FACT_PATTERNS = {
    "Q01": [
        ["bart large", "bart generator"],
        ["400m"],
        ["sequence to sequence", "seq2seq"],
    ],
    "Q03": [
        ["same retrieved document", "same document", "sequence probability for each document"],
        ["single latent variable", "latent document"],
        ["different latent document for each target token", "different latent document", "per token"],
    ],
    "Q05": [
        ["extract a set of statements", "decompose longer sentences", "atomic statements"],
        ["verification function", "inferred from", "supported by the information"],
        ["final faithfulness score", "fraction", "|v|", "|s|"],
    ],
    "Q06": [
        ["44.1"], ["44.5"], ["41.5"], ["57.9"], ["55.2"], ["66.1"], ["68.0"],
    ],
}


def norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def record(chunk, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "physical_page_number": chunk.physical_page_number,
        "printed_page_number": chunk.printed_page_number,
        "score": chunk.score,
        "retrieval_method": chunk.retrieval_method,
        "dense_rank": chunk.dense_rank,
        "bm25_rank": chunk.bm25_rank,
        "dense_score": chunk.dense_score,
        "bm25_score": chunk.bm25_score,
        "text": chunk.text,
    }


def fact_hits(facts, chunks):
    texts = [norm(chunk["text"]) for chunk in chunks]
    result = []
    for alternatives in facts:
        hits = [idx + 1 for idx, text in enumerate(texts) if any(norm(pattern) in text for pattern in alternatives)]
        result.append({"alternatives": alternatives, "supported": bool(hits), "ranks": hits})
    return result


def main() -> int:
    questions = {item["id"]: item for item in json.loads((ROOT / "evaluation/questions/answerable_questions.json").read_text(encoding="utf-8"))}
    dense_diagnostic = json.loads(DENSE_DIAGNOSTIC.read_text(encoding="utf-8"))
    dense_diag_by_id = {item["id"]: item for item in dense_diagnostic["questions"]}
    dense = DenseRetriever()
    # Share the dense client with HybridRetriever so local Qdrant's single
    # process storage lock is not contended by two independent clients.
    hybrid = HybridRetriever(dense_retriever=dense)
    output = {
        "diagnostic": "dense and hybrid retrieval only; no generation/provider calls",
        "top_k": 5,
        "question_ids": QUESTION_IDS,
        "questions": [],
    }
    try:
        for qid in QUESTION_IDS:
            question = questions[qid]
            dense_result = dense.retrieve(question["question"], top_k=5)
            hybrid_result = hybrid.retrieve(question["question"], top_k=5)
            dense_chunks = [record(chunk, rank) for rank, chunk in enumerate(dense_result.chunks, 1)]
            hybrid_chunks = [record(chunk, rank) for rank, chunk in enumerate(hybrid_result.chunks, 1)]
            dense_facts = fact_hits(FACT_PATTERNS[qid], dense_chunks)
            hybrid_facts = fact_hits(FACT_PATTERNS[qid], hybrid_chunks)
            dense_count = sum(1 for fact in dense_facts if fact["supported"])
            hybrid_count = sum(1 for fact in hybrid_facts if fact["supported"])
            def coverage(count):
                return "complete" if count == len(FACT_PATTERNS[qid]) else "partial" if count else "missing"
            dense_diag = dense_diag_by_id[qid]
            deep_target_hits = [
                {"rank": chunk["rank"], "chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"], "physical_page_number": chunk["physical_page_number"], "score": chunk["score"]}
                for chunk in dense_diag["diagnostic_search_top_50"]
                if chunk["document_id"] in question.get("source_papers", [])
            ]
            output["questions"].append({
                "id": qid,
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "source_papers": question.get("source_papers", []),
                "page_references": question.get("page_references", ""),
                "required_fact_signals": FACT_PATTERNS[qid],
                "dense": {
                    "coverage": coverage(dense_count),
                    "fact_signals": dense_facts,
                    "execution_time_ms": dense_result.execution_time_ms,
                    "top_five": dense_chunks,
                },
                "hybrid": {
                    "coverage": coverage(hybrid_count),
                    "fact_signals": hybrid_facts,
                    "execution_time_ms": hybrid_result.execution_time_ms,
                    "top_five": hybrid_chunks,
                },
                "dense_existing_diagnostic_target_hits_top100": deep_target_hits,
            })
    finally:
        dense.close()
        hybrid.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)
    for item in output["questions"]:
        print(f"{item['id']}: dense={item['dense']['coverage']}; hybrid={item['hybrid']['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
