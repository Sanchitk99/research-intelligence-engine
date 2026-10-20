"""Offline tests for the opt-in CPU cross-encoder reranker."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import (
    CpuCrossEncoderReranker,
    RerankerModelNotCachedError,
    RerankerResourceError,
    is_model_cached,
    model_cache_directory,
    require_resources,
)


def chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="P1", document_title="Paper", physical_page_index=2,
        physical_page_number=3, printed_page_number=3, chunk_index_in_page=0,
        chunk_index_in_doc=0, start_char_idx=0, end_char_idx=len(text), char_count=len(text),
        word_count=len(text.split()), text=text, score=0.1, retrieval_method="dense",
    )


def cached_model(tmp_path: Path, model: str = "cross-encoder/ms-marco-MiniLM-L2-v2") -> Path:
    snap = model_cache_directory(model, tmp_path) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"weights")
    return snap


def test_cache_detection(tmp_path):
    assert not is_model_cached(cache_root=tmp_path)
    cached_model(tmp_path)
    assert is_model_cached(cache_root=tmp_path)


def test_missing_cache_fails_before_model_load(tmp_path):
    with pytest.raises(RerankerModelNotCachedError):
        CpuCrossEncoderReranker(cache_root=tmp_path, cross_encoder_factory=lambda *args, **kwargs: pytest.fail("loaded"))


def test_deduplication_order_ties_and_metadata_preservation(tmp_path):
    cached_model(tmp_path)

    class FakeCrossEncoder:
        def __init__(self, model, device):
            assert device == "cpu"
        def predict(self, pairs, batch_size, show_progress_bar):
            return [1.0 for _ in pairs]

    first = chunk("c1", "first")
    duplicate = chunk("c1", "duplicate metadata must not replace first")
    second = chunk("c2", "second")
    reranker = CpuCrossEncoderReranker(cache_root=tmp_path, cross_encoder_factory=FakeCrossEncoder)
    result = reranker.rerank("question", [first, duplicate, second], top_k=2)
    assert result.candidate_count == 2
    assert [c.chunk_id for c in result.chunks] == ["c1", "c2"]
    assert result.chunks[0] is first
    assert result.scores == [1.0, 1.0]


def test_scores_rank_deterministically(tmp_path):
    cached_model(tmp_path)

    class FakeCrossEncoder:
        def __init__(self, model, device): pass
        def predict(self, pairs, batch_size, show_progress_bar): return [0.2, 0.9, 0.5]

    reranker = CpuCrossEncoderReranker(cache_root=tmp_path, cross_encoder_factory=FakeCrossEncoder)
    result = reranker.rerank("q", [chunk("a", "a"), chunk("b", "b"), chunk("c", "c")])
    assert [c.chunk_id for c in result.chunks] == ["b", "c", "a"]


def test_resource_guard():
    with pytest.raises(RerankerResourceError):
        require_resources(min_ram_mb=10**12)


def test_default_output_paths_are_experiment_specific():
    from scripts.run_cross_encoder_reranking_evaluation import parse_args

    args = parse_args([])
    assert "reranking_evaluation_cpu_minilm_l2" in args.output_json.name
    assert "reranking_evaluation_cpu_minilm_l2" in args.output_report.name
    assert args.output_json.name not in {"retrieval_evaluation.json", "hybrid_retrieval_evaluation.json"}

