import json

import pytest

from src.evaluation.groq_generation_runner import (
    build_run_config,
    CallBudgetExceeded,
    CallBudget,
    CheckpointMismatchError,
    CheckpointStore,
    atomic_write_json,
)


def config(mode="dense", model="openai/gpt-oss-20b", budget=28):
    return {
        "provider": "groq",
        "model": model,
        "retrieval_mode": mode,
        "top_k": 5,
        "max_call_budget": budget,
        "warmup_calls": 0,
        "question_ids": ["Q01", "Q02", "OOS01"],
        "answerable_questions_sha256": "a",
        "out_of_scope_questions_sha256": "b",
        "prompt_sha256": "p",
        "scoring_sha256": "s",
        "scoring_version": "generation-evaluator-v1",
    }


def result(question_id="Q01", call=True):
    return {
        "question_id": question_id,
        "question_text": "question",
        "category": "answerable",
        "expected_outcome": "answer",
        "status": "answered",
        "answer": "answer",
        "appropriate_outcome": True,
        "generation_call_made": call,
        "generation_call_succeeded": call,
    }


def test_checkpoint_creation_and_atomic_update(tmp_path):
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(path, config())
    store.mark_complete("Q01", result(), provider_call_made=True)
    assert json.loads(path.read_text(encoding="utf-8"))["questions"]["Q01"]["state"] == "complete"
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_skips_completed_but_not_failed(tmp_path):
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(path, config())
    store.mark_complete("Q01", result(), provider_call_made=True)
    store.mark_failed("Q02", {"type": "RateLimitError", "message": "quota"}, provider_call_made=True)
    resumed = CheckpointStore.open(path, config(), resume=True)
    assert resumed.completed_ids() == {"Q01"}
    assert resumed.counts()["failed_questions"] == 1
    assert resumed.counts()["remaining_questions"] == 1


def test_dense_and_hybrid_checkpoints_are_not_mixable(tmp_path):
    dense_path = tmp_path / "dense.json"
    CheckpointStore.open(dense_path, config("dense"))
    with pytest.raises(CheckpointMismatchError):
        CheckpointStore.open(dense_path, config("hybrid"), resume=True)


def test_failed_questions_are_not_complete(tmp_path):
    store = CheckpointStore.open(tmp_path / "checkpoint.json", config())
    store.mark_failed("Q01", {"type": "NetworkError", "message": "offline"}, provider_call_made=False)
    assert "Q01" not in store.completed_ids()


def test_call_budget_enforcement():
    budget = CallBudget(1, 1)
    with pytest.raises(CallBudgetExceeded):
        budget.ensure_available()


def test_interruption_preserves_completed_progress(tmp_path):
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(path, config())
    store.mark_complete("Q01", result(), provider_call_made=True)
    store.mark_status("interrupted", "KeyboardInterrupt")
    resumed = CheckpointStore.open(path, config(), resume=True)
    assert resumed.completed_ids() == {"Q01"}
    assert resumed.payload["status"] == "interrupted"


def test_progress_counts_include_skips_and_attempts(tmp_path):
    store = CheckpointStore.open(tmp_path / "checkpoint.json", config())
    store.mark_complete("Q01", result(call=False), provider_call_made=False)
    store.mark_failed("Q02", {"type": "Quota", "message": "limit"}, provider_call_made=True)
    counts = store.counts()
    assert counts["completed_questions"] == 1
    assert counts["failed_questions"] == 1
    assert counts["remaining_questions"] == 1
    assert counts["skipped_provider_calls"] == 1
    assert counts["provider_calls_used"] == 1
    assert counts["provider_call_attempts"] == 1


def test_atomic_write_replaces_existing_file(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 2


def test_run_fingerprint_records_schema_tokens_and_reasoning(monkeypatch):
    monkeypatch.setenv("GROQ_MAX_COMPLETION_TOKENS", "3072")
    run_config = build_run_config("dense", "openai/gpt-oss-20b", 5, 28)
    assert run_config["model"] == "openai/gpt-oss-20b"
    assert run_config["response_schema_version"] == "research-intelligence-answer-v1"
    assert run_config["max_completion_tokens"] == 3072
    assert run_config["reasoning_effort"] == "medium"
    assert run_config["reasoning_format"] is None
