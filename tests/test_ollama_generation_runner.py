"""Offline safety tests for the resumable Ollama evaluation runner."""

import json
from pathlib import Path

import pytest

from src.evaluation.groq_generation_runner import CheckpointMismatchError, CheckpointStore
from src.evaluation.ollama_generation_runner import (
    OLLAMA_SCHEMA_VERSION,
    build_run_config,
    checkpoint_path,
    final_paths,
    incomplete_paths,
)


def _config(tmp_path: Path) -> dict:
    return {
        "provider": "ollama",
        "model": "gemma3:4b",
        "base_url": "http://localhost:11434/v1",
        "retrieval_mode": "dense",
        "top_k": 5,
        "max_call_budget": 28,
        "question_ids": ["Q01", "Q02", "OOS01"],
        "prompt_sha256": "prompt",
        "scoring_sha256": "scoring",
    }


def test_checkpoint_creation_is_atomic_and_has_ollama_schema(tmp_path):
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(path, _config(tmp_path), schema_version=OLLAMA_SCHEMA_VERSION, failed_are_retryable=True)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == OLLAMA_SCHEMA_VERSION
    assert not list(tmp_path.glob("*.tmp"))

    store.mark_complete("Q01", {"question_id": "Q01"}, provider_call_made=True)
    reloaded = CheckpointStore.open(path, _config(tmp_path), resume=True, schema_version=OLLAMA_SCHEMA_VERSION, failed_are_retryable=True)
    assert reloaded.completed_ids() == {"Q01"}
    assert reloaded.payload["questions"]["Q01"]["result"]["question_id"] == "Q01"


def test_failed_question_is_preserved_and_retryable_on_resume(tmp_path):
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(path, _config(tmp_path), schema_version=OLLAMA_SCHEMA_VERSION, failed_are_retryable=True)
    store.mark_complete("Q01", {"question_id": "Q01"}, provider_call_made=True)
    store.mark_failed("Q02", {"type": "ProviderError", "message": "local failure"}, provider_call_made=True)
    resumed = CheckpointStore.open(path, _config(tmp_path), resume=True, schema_version=OLLAMA_SCHEMA_VERSION, failed_are_retryable=True)
    counts = resumed.counts()
    assert resumed.completed_ids() == {"Q01"}
    assert resumed.payload["questions"]["Q02"]["state"] == "failed"
    assert counts["completed_questions"] == 1
    assert counts["failed_questions"] == 1
    assert counts["remaining_questions"] == 2


def test_configuration_mismatch_rejected_before_resume(tmp_path):
    path = tmp_path / "checkpoint.json"
    config = _config(tmp_path)
    CheckpointStore.open(path, config, schema_version=OLLAMA_SCHEMA_VERSION, failed_are_retryable=True)
    changed = dict(config, model="gemma3:latest")
    with pytest.raises(CheckpointMismatchError):
        CheckpointStore.open(path, changed, resume=True, schema_version=OLLAMA_SCHEMA_VERSION, failed_are_retryable=True)


def test_dense_and_hybrid_and_final_paths_are_ollama_specific():
    assert checkpoint_path("dense").name == "generation_evaluation_ollama_dense_checkpoint.json"
    assert checkpoint_path("hybrid").name == "generation_evaluation_ollama_hybrid_checkpoint.json"
    assert all("ollama" in path.name for path in (*final_paths("dense"), *incomplete_paths("hybrid")))
    assert all("generation_evaluation.json" not in path.name for path in (*final_paths("dense"), *final_paths("hybrid")))


def test_run_config_contains_provider_model_and_generation_fingerprint():
    config = build_run_config("hybrid", "gemma3:4b", 5, 28)
    assert config["provider"] == "ollama"
    assert config["model"] == "gemma3:4b"
    assert config["retrieval_mode"] == "hybrid"
    assert config["base_url"] == "http://localhost:11434/v1"
    assert config["generation_config"]["response_format"] == "json_object"
    assert config["generation_config"]["max_tokens"] == 1024
