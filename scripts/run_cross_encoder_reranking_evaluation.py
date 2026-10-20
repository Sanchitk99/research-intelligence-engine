"""Opt-in, CPU-only cross-encoder retrieval evaluation.

This script never downloads a model and never changes the default retrievers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.ground_truth import GroundTruthTarget, load_ground_truth
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    CpuCrossEncoderReranker,
    RerankerError,
    active_ollama_models,
    available_ram_mb,
    is_model_cached,
    model_cache_directory,
    require_resources,
)
from src.retrieval.retriever import DenseRetriever


class MappingRetriever:
    """Adapter allowing the existing RetrievalEvaluator to score precomputed results."""

    def __init__(self, results: Dict[str, RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        result = self.results[query]
        return result.model_copy(update={"chunks": result.chunks[:top_k], "total_retrieved": min(top_k, len(result.chunks))})


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate opt-in CPU cross-encoder reranking.")
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--questions-file", type=Path, default=Path("evaluation/questions/answerable_questions.json"))
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-ram-mb", type=int, default=3072)
    parser.add_argument("--output-json", type=Path, default=Path("evaluation/results/reranking_evaluation_cpu_minilm_l2.json"))
    parser.add_argument("--output-report", type=Path, default=Path("evaluation/results/reranking_evaluation_cpu_minilm_l2.md"))
    return parser.parse_args(args)


def _result(query: str, chunks, elapsed_ms: float) -> RetrievalResult:
    return RetrievalResult(query=query, chunks=chunks, total_retrieved=len(chunks), collection_name="experiment", execution_time_ms=round(elapsed_ms, 2))


def _write_blocked(path: Path, parsed, reason: str, ram_mb, ollama_models) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked_before_inference",
        "model": parsed.model,
        "device": "cpu",
        "candidate_k_per_retriever": parsed.candidate_k,
        "final_top_k": parsed.top_k,
        "output_json": str(parsed.output_json),
        "output_report": str(parsed.output_report),
        "model_cache_directory": str(model_cache_directory(parsed.model)),
        "model_cached": is_model_cached(parsed.model),
        "available_ram_mb": ram_mb,
        "ollama_loaded_models": ollama_models,
        "reason": reason,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(args=None) -> int:
    parsed = parse_args(args)
    print(f"Model: {parsed.model}\nDevice: cpu\nCandidate limit: {parsed.candidate_k * 2}\nOutput: {parsed.output_json}")
    ram_mb = available_ram_mb()
    ollama_models = active_ollama_models()
    try:
        if not is_model_cached(parsed.model):
            reason = f"Model '{parsed.model}' is not available in the local Hugging Face cache; no download attempted"
            _write_blocked(parsed.output_json, parsed, reason, ram_mb, ollama_models)
            print(reason, file=sys.stderr)
            return 2
        require_resources(parsed.min_ram_mb)
    except RerankerError as exc:
        _write_blocked(parsed.output_json, parsed, str(exc), ram_mb, ollama_models)
        print(str(exc), file=sys.stderr)
        return 2

    targets = load_ground_truth(parsed.questions_file)
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    try:
        reranker = CpuCrossEncoderReranker(parsed.model, parsed.batch_size, parsed.candidate_k * 2)
        dense_results: Dict[str, RetrievalResult] = {}
        hybrid_results: Dict[str, RetrievalResult] = {}
        reranked_results: Dict[str, RetrievalResult] = {}
        details: List[dict] = []
        for target in targets:
            query = target.question_text
            d = dense.retrieve(query, top_k=parsed.candidate_k)
            b = bm25.retrieve(query, top_k=parsed.candidate_k)
            fused = reciprocal_rank_fusion(d.chunks, b.chunks, top_k=parsed.top_k)
            union = list(d.chunks) + [chunk for chunk in b.chunks if chunk.chunk_id not in {c.chunk_id for c in d.chunks}]
            rr = reranker.rerank(query, union, top_k=parsed.top_k)
            dense_results[query] = _result(query, d.chunks, d.execution_time_ms)
            hybrid_results[query] = _result(query, fused, d.execution_time_ms + b.execution_time_ms)
            reranked_results[query] = _result(query, rr.chunks, d.execution_time_ms + b.execution_time_ms + rr.scoring_latency_ms)
            details.append({"question_id": target.question_id, "candidate_union_size": rr.candidate_count, "reranker_scoring_latency_ms": rr.scoring_latency_ms, "dense_latency_ms": d.execution_time_ms, "bm25_latency_ms": b.execution_time_ms, "hybrid_latency_ms": round(d.execution_time_ms + b.execution_time_ms, 2), "reranked_latency_ms": round(d.execution_time_ms + b.execution_time_ms + rr.scoring_latency_ms, 2), "reranker_scores": rr.scores, "reranked_chunk_ids": [c.chunk_id for c in rr.chunks]})
        summaries = {}
        for name, results in (("dense", dense_results), ("hybrid", hybrid_results), ("reranked", reranked_results)):
            summary, question_results = RetrievalEvaluator(MappingRetriever(results), default_top_k=parsed.top_k).evaluate_all(targets, total_corpus_questions=len(targets), excluded_count=10, top_k=parsed.top_k)
            summaries[name] = {"summary": summary.model_dump(), "questions": [r.model_dump() for r in question_results]}
        output = {"status": "complete", "model": parsed.model, "device": "cpu", "candidate_k_per_retriever": parsed.candidate_k, "final_top_k": parsed.top_k, "answerable_questions": len(targets), "out_of_scope_questions_excluded": 10, "available_ram_before_model_mb": ram_mb, "ollama_loaded_models": ollama_models, "details": details, "modes": summaries}
        parsed.output_json.parent.mkdir(parents=True, exist_ok=True)
        parsed.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        parsed.output_report.write_text("# CPU Cross-Encoder Reranking Evaluation\n\nSee the machine-readable artifact for full per-question results.\n", encoding="utf-8")
        print(parsed.output_json)
        return 0
    finally:
        dense.close()


if __name__ == "__main__":
    raise SystemExit(main())

