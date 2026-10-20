"""Resumable local Ollama generation evaluation.

This runner reuses the existing question loading, retrieval planning, checkpoint,
budget, evaluator, and reporting abstractions.  It owns Ollama-specific paths and
fingerprints so it cannot mix with Groq or historical benchmark artifacts.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx

from src.evaluation.generation_evaluator import (
    GenerationEvaluationSummary,
    GenerationEvaluator,
    QuestionGenEvalResult,
    save_generation_results,
)
from src.evaluation.groq_generation_runner import (
    ANSWERABLE_PATH,
    OOS_PATH,
    PROMPTS_PATH,
    RESULTS_DIR,
    SCORING_PATH,
    CallBudget,
    CheckpointError,
    CheckpointMismatchError,
    CheckpointStore,
    BudgetedGenerator,
    _retrieval_plan,
    _temporary_question_file,
    atomic_write_json,
    load_questions,
    question_order,
    sha256_file,
)
from src.generation.generator import AnswerGenerator
from src.generation.models import AnswerStatus, GenerationConfig
from src.generation.providers.factory import get_llm_provider


OLLAMA_SCHEMA_VERSION = "ollama-generation-checkpoint-v1"
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MAX_CALLS = 28
OLLAMA_GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 1024,
    "max_completion_tokens": None,
    "timeout_seconds": 30.0,
    "response_format": "json_object",
}


def checkpoint_path(mode: str) -> Path:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    return RESULTS_DIR / f"generation_evaluation_ollama_{mode}_checkpoint.json"


def final_paths(mode: str) -> tuple[Path, Path]:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    return (
        RESULTS_DIR / f"generation_evaluation_ollama_{mode}.json",
        RESULTS_DIR / f"generation_evaluation_ollama_{mode}_report.md",
    )


def incomplete_paths(mode: str) -> tuple[Path, Path]:
    return (
        RESULTS_DIR / f"generation_evaluation_ollama_{mode}_incomplete.json",
        RESULTS_DIR / f"generation_evaluation_ollama_{mode}_incomplete.md",
    )


def _local_base_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return clean[:-3] if clean.lower().endswith("/v1") else clean


def verify_local_model(model: str, base_url: str) -> Dict[str, Any]:
    """Verify the explicitly selected model through Ollama's localhost API."""
    try:
        response = httpx.get(f"{_local_base_url(base_url)}/api/tags", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise CheckpointError(f"Ollama local API unavailable: {exc}") from exc
    models = [str(item.get("name")) for item in payload.get("models", []) if isinstance(item, dict)]
    if model not in models:
        raise CheckpointError(f"Configured Ollama model '{model}' is not installed; available models: {models}")
    return {"base_url": base_url, "model": model, "available_models": models}


def build_run_config(mode: str, model: str, top_k: int, max_calls: int, base_url: Optional[str] = None) -> Dict[str, Any]:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    endpoint = (base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)).rstrip("/")
    return {
        "provider": "ollama",
        "model": model,
        "base_url": endpoint,
        "retrieval_mode": mode,
        "top_k": top_k,
        "max_call_budget": max_calls,
        "warmup_calls": 0,
        "question_ids": question_order(),
        "answerable_questions_sha256": sha256_file(ANSWERABLE_PATH),
        "out_of_scope_questions_sha256": sha256_file(OOS_PATH),
        "prompt_sha256": sha256_file(PROMPTS_PATH),
        "scoring_sha256": sha256_file(SCORING_PATH),
        "scoring_version": "generation-evaluator-v1",
        "response_schema_version": "application-json-object-v1",
        "generation_config": OLLAMA_GENERATION_CONFIG,
    }


def _validate_paths(mode: str) -> None:
    paths = [checkpoint_path(mode), *final_paths(mode), *incomplete_paths(mode)]
    forbidden = {
        RESULTS_DIR / "generation_evaluation.json",
        RESULTS_DIR / "generation_evaluation_dense.json",
        RESULTS_DIR / "generation_evaluation_hybrid.json",
        RESULTS_DIR / "generation_evaluation_report.md",
        RESULTS_DIR / "generation_evaluation_dense_report.md",
        RESULTS_DIR / "generation_evaluation_hybrid_report.md",
    }
    if any(path in forbidden for path in paths):
        raise CheckpointError("Ollama output path overlaps a historical/generic evaluation artifact")


def _validate_existing_checkpoint(mode: str, config: Dict[str, Any]) -> str:
    path = checkpoint_path(mode)
    if not path.exists():
        return "not_present_new_run"
    CheckpointStore.open(
        path,
        config,
        resume=True,
        schema_version=OLLAMA_SCHEMA_VERSION,
        failed_are_retryable=True,
    )
    return "compatible_existing_checkpoint"


def preflight(mode: str, model: str, top_k: int, max_calls: int) -> Dict[str, Any]:
    config = build_run_config(mode, model, top_k, max_calls)
    _validate_paths(mode)
    provider = get_llm_provider(provider_type="ollama", model_name=model)
    if provider.name != "ollama" or provider.default_model != model:
        raise CheckpointError("Provider factory did not return the requested Ollama provider/model")
    availability = verify_local_model(model, config["base_url"])
    checkpoint_state = _validate_existing_checkpoint(mode, config)
    plan = _retrieval_plan(mode, top_k)
    expected = sum(1 for will_call in plan.values() if will_call)
    report = {
        "status": "preflight_only",
        "provider": "ollama",
        "model": model,
        "retrieval_mode": mode,
        "question_count": len(plan),
        "retrieval_nonempty_questions": sum(plan.values()),
        "retrieval_empty_questions": len(plan) - sum(plan.values()),
        "expected_provider_calls": expected,
        "max_call_budget": max_calls,
        "budget_sufficient_for_current_retrieval_plan": max_calls >= expected,
        "question_ids_skipping_generation": [qid for qid, will_call in plan.items() if not will_call],
        "generation_calls_made": 0,
        "local_model_availability": availability,
        "checkpoint_state": checkpoint_state,
        "configuration": config,
        "output_paths": {
            "checkpoint": str(checkpoint_path(mode)),
            "json": str(final_paths(mode)[0]),
            "markdown": str(final_paths(mode)[1]),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_DIR / f"generation_evaluation_ollama_{mode}_preflight.json", report)
    return report


def _incomplete_artifacts(mode: str, checkpoint: CheckpointStore) -> None:
    json_path, md_path = incomplete_paths(mode)
    atomic_write_json(json_path, checkpoint.payload)
    counts = checkpoint.counts()
    md_path.write_text(
        f"# Ollama Generation Evaluation — {mode.title()} — Incomplete\n\n"
        f"- Provider/model: `ollama` / `{checkpoint.config['model']}`\n"
        f"- Completed: {counts['completed_questions']} / {counts['total_questions']}\n"
        f"- Failed: {counts['failed_questions']}\n"
        f"- Remaining retryable: {counts['remaining_questions']}\n"
        f"- Provider calls used: {counts['provider_calls_used']} / {checkpoint.config['max_call_budget']}\n"
        f"- Checkpoint: `{checkpoint.path}`\n",
        encoding="utf-8",
    )


def run_mode(mode: str, model: str, top_k: int, max_calls: int, resume: bool = False) -> Dict[str, Any]:
    config = build_run_config(mode, model, top_k, max_calls)
    _validate_paths(mode)
    checkpoint = CheckpointStore.open(
        checkpoint_path(mode),
        config,
        resume=resume,
        schema_version=OLLAMA_SCHEMA_VERSION,
        failed_are_retryable=True,
    )
    checkpoint.mark_status("running")
    answerable, out_of_scope = load_questions()
    items = [(item, "answerable") for item in answerable] + [(item, "out_of_scope") for item in out_of_scope]
    retrieval_plan = _retrieval_plan(mode, top_k)
    call_by_question = {item["question"]: retrieval_plan.get(item["id"], True) for item in answerable + out_of_scope}
    verify_local_model(model, config["base_url"])
    provider = get_llm_provider(provider_type="ollama", model_name=model)
    if provider.name != "ollama" or provider.default_model != model:
        raise CheckpointError("Explicit Ollama provider/model verification failed")
    generator = AnswerGenerator(
        provider=provider,
        config=GenerationConfig(
            provider_type="ollama",
            model_name=model,
            retrieval_mode=mode,
            top_k=top_k,
            temperature=OLLAMA_GENERATION_CONFIG["temperature"],
            max_tokens=OLLAMA_GENERATION_CONFIG["max_tokens"],
            timeout_seconds=OLLAMA_GENERATION_CONFIG["timeout_seconds"],
        ),
        retrieval_mode=mode,
    )
    budget = CallBudget(checkpoint.counts()["provider_calls_used"], max_calls)
    budgeted_generator = BudgetedGenerator(generator, budget, call_by_question)
    try:
        for item, category in items:
            qid = item["id"]
            if qid in checkpoint.completed_ids():
                continue
            try:
                with tempfile.TemporaryDirectory(prefix=f"ollama_eval_{mode}_") as temp_dir:
                    temp = Path(temp_dir)
                    ans_path = _temporary_question_file(temp, "answerable.json", [item] if category == "answerable" else [])
                    oos_path = _temporary_question_file(temp, "oos.json", [item] if category == "out_of_scope" else [])
                    evaluator = GenerationEvaluator(generator=budgeted_generator)
                    _, results = evaluator.evaluate_all(
                        answerable_path=ans_path,
                        out_of_scope_path=oos_path,
                        top_k=top_k,
                        warmup=False,
                        cold_start_measurement_mode="standalone",
                        fail_fast_on_generation_error=False,
                    )
                result = results[0]
                call_made = bool(result.generation_call_made)
                if result.status == AnswerStatus.ERROR or (call_made and not result.generation_call_succeeded):
                    checkpoint.mark_failed(
                        qid,
                        {"type": result.failure_type or "GenerationError", "stage": result.failure_stage or "generation", "message": result.failure_message or result.confidence_notes or result.answer},
                        provider_call_made=call_made,
                    )
                    checkpoint.mark_status("failed")
                    _incomplete_artifacts(mode, checkpoint)
                    return checkpoint.payload
                checkpoint.mark_complete(qid, result.model_dump(mode="json"), provider_call_made=call_made)
            except KeyboardInterrupt:
                checkpoint.mark_status("interrupted", "KeyboardInterrupt")
                _incomplete_artifacts(mode, checkpoint)
                raise
            except Exception as exc:
                checkpoint.mark_failed(qid, {"type": type(exc).__name__, "stage": "evaluation", "message": str(exc)}, provider_call_made=False)
                checkpoint.mark_status("failed", str(exc))
                _incomplete_artifacts(mode, checkpoint)
                return checkpoint.payload

        completed_results = [
            QuestionGenEvalResult.model_validate(entry["result"])
            for entry in checkpoint.payload["questions"].values()
            if entry.get("state") == "complete"
        ]
        evaluator = GenerationEvaluator(generator=budgeted_generator)
        summary: GenerationEvaluationSummary = evaluator.summarize_results(completed_results, cold_start_measurement_mode="standalone")
        final_json, final_md = final_paths(mode)
        save_generation_results(summary, completed_results, final_json, final_md)
        checkpoint.mark_status("complete")
        checkpoint.payload["final_artifacts"] = {"json": str(final_json), "markdown": str(final_md)}
        checkpoint.save()
        return checkpoint.payload
    finally:
        generator.close()
