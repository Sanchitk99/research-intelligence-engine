"""Resumable, quota-aware evaluation runner for the Groq provider.

This module intentionally keeps the existing evaluator as the source of scoring
truth.  It adds only run orchestration, checkpointing, and configuration
fingerprints; it does not change retrieval, prompts, or scoring rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.evaluation.generation_evaluator import (
    GenerationEvaluationSummary,
    GenerationEvaluator,
    QuestionGenEvalResult,
    save_generation_results,
)
from src.generation.generator import AnswerGenerator
from src.generation.models import AnswerStatus, GenerationConfig
from src.generation.providers.factory import get_llm_provider
from src.generation.providers.groq_provider import (
    GROQ_DEFAULT_MAX_COMPLETION_TOKENS,
    GROQ_REASONING_EFFORT,
    GROQ_REASONING_FORMAT,
    GROQ_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
ANSWERABLE_PATH = ROOT / "evaluation" / "questions" / "answerable_questions.json"
OOS_PATH = ROOT / "evaluation" / "questions" / "out_of_scope_questions.json"
PROMPTS_PATH = ROOT / "src" / "generation" / "prompts.py"
SCORING_PATH = ROOT / "src" / "evaluation" / "generation_evaluator.py"
RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_MODEL = "openai/gpt-oss-20b"
SCHEMA_VERSION = "groq-generation-checkpoint-v1"
SCORING_VERSION = "generation-evaluator-v1"


class CheckpointError(RuntimeError):
    """Base class for unsafe or invalid checkpoint operations."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a checkpoint was made for a different experiment."""


class CallBudgetExceeded(CheckpointError):
    """Raised when another provider call would exceed the configured budget."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    answerable = json.loads(ANSWERABLE_PATH.read_text(encoding="utf-8"))
    out_of_scope = json.loads(OOS_PATH.read_text(encoding="utf-8"))
    if len(answerable) != 18 or len(out_of_scope) != 10:
        raise CheckpointError(
            f"Unexpected evaluation set size: answerable={len(answerable)}, out_of_scope={len(out_of_scope)}"
        )
    ids = [item["id"] for item in answerable + out_of_scope]
    if len(ids) != len(set(ids)):
        raise CheckpointError("Evaluation question IDs are not unique")
    return answerable, out_of_scope


def question_order() -> List[str]:
    answerable, out_of_scope = load_questions()
    return [item["id"] for item in answerable + out_of_scope]


def checkpoint_path(mode: str) -> Path:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    return RESULTS_DIR / f"generation_evaluation_groq_{mode}_checkpoint.json"


def build_run_config(mode: str, model: str, top_k: int, max_calls: int) -> Dict[str, Any]:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    return {
        "provider": "groq",
        "model": model,
        "retrieval_mode": mode,
        "top_k": top_k,
        "max_call_budget": max_calls,
        "warmup_calls": 0,
        "question_ids": question_order(),
        "answerable_questions_sha256": sha256_file(ANSWERABLE_PATH),
        "out_of_scope_questions_sha256": sha256_file(OOS_PATH),
        "prompt_sha256": sha256_file(PROMPTS_PATH),
        "scoring_sha256": sha256_file(SCORING_PATH),
        "scoring_version": SCORING_VERSION,
        "response_schema_version": GROQ_SCHEMA_VERSION,
        "max_completion_tokens": int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", str(GROQ_DEFAULT_MAX_COMPLETION_TOKENS))),
        "reasoning_effort": GROQ_REASONING_EFFORT,
        "reasoning_format": GROQ_REASONING_FORMAT,
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist JSON with fsync and replace, leaving no partial target file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


class CheckpointStore:
    """Atomic checkpoint store whose only resumable state is a complete result."""

    def __init__(self, path: Path, config: Dict[str, Any], payload: Dict[str, Any], *, schema_version: str = SCHEMA_VERSION, failed_are_retryable: bool = False):
        self.path = path
        self.config = config
        self.payload = payload
        self.schema_version = schema_version
        self.failed_are_retryable = failed_are_retryable

    @classmethod
    def open(
        cls,
        path: Path,
        config: Dict[str, Any],
        resume: bool = False,
        *,
        schema_version: str = SCHEMA_VERSION,
        failed_are_retryable: bool = False,
    ) -> "CheckpointStore":
        if path.exists():
            if not resume:
                raise CheckpointError(f"Checkpoint exists; pass --resume to continue: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc
            if payload.get("schema_version") != schema_version:
                raise CheckpointMismatchError("Checkpoint schema version is incompatible")
            if payload.get("run_config") != config:
                raise CheckpointMismatchError("Checkpoint configuration does not match this run")
            if not isinstance(payload.get("questions"), dict):
                raise CheckpointMismatchError("Checkpoint questions section is invalid")
            return cls(path, config, payload, schema_version=schema_version, failed_are_retryable=failed_are_retryable)
        payload = {
            "schema_version": schema_version,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "initialized",
            "run_config": config,
            "question_order": config["question_ids"],
            "questions": {},
            "progress": {
                "total_questions": len(config["question_ids"]),
                "completed_questions": 0,
                "failed_questions": 0,
                "remaining_questions": len(config["question_ids"]),
                "provider_calls_used": 0,
                "provider_calls_remaining": config["max_call_budget"],
                "provider_call_attempts": 0,
                "skipped_provider_calls": 0,
            },
        }
        store = cls(path, config, payload, schema_version=schema_version, failed_are_retryable=failed_are_retryable)
        store.save()
        return store

    def save(self) -> None:
        self.payload["updated_at"] = _utc_now()
        atomic_write_json(self.path, self.payload)

    def _refresh_progress(self) -> Dict[str, int]:
        entries = self.payload["questions"].values()
        complete = sum(1 for entry in entries if entry.get("state") == "complete")
        failed = sum(1 for entry in self.payload["questions"].values() if entry.get("state") == "failed")
        calls = sum(1 for entry in self.payload["questions"].values() if entry.get("provider_call_made"))
        attempts = sum(1 for entry in self.payload["questions"].values() if entry.get("provider_call_attempted"))
        skipped = sum(
            1 for entry in self.payload["questions"].values()
            if entry.get("state") == "complete" and not entry.get("provider_call_made")
        )
        total = len(self.config["question_ids"])
        remaining = total - complete if self.failed_are_retryable else total - complete - failed
        self.payload["progress"] = {
            "total_questions": total,
            "completed_questions": complete,
            "failed_questions": failed,
            "remaining_questions": remaining,
            "provider_calls_used": calls,
            "provider_calls_remaining": self.config["max_call_budget"] - calls,
            "provider_call_attempts": attempts,
            "skipped_provider_calls": skipped,
        }
        return self.payload["progress"]

    def completed_ids(self) -> set[str]:
        return {qid for qid, entry in self.payload["questions"].items() if entry.get("state") == "complete"}

    def counts(self) -> Dict[str, int]:
        return dict(self._refresh_progress())

    def mark_complete(self, question_id: str, result: Dict[str, Any], provider_call_made: bool) -> None:
        self.payload["questions"][question_id] = {
            "state": "complete",
            "question_id": question_id,
            "provider": self.config["provider"],
            "model": self.config["model"],
            "provider_call_attempted": provider_call_made,
            "provider_call_made": provider_call_made,
            "completed_at": _utc_now(),
            "result": result,
        }
        self._refresh_progress()
        self.save()

    def mark_failed(self, question_id: str, error: Dict[str, Any], provider_call_made: bool) -> None:
        self.payload["questions"][question_id] = {
            "state": "failed",
            "question_id": question_id,
            "provider": self.config["provider"],
            "model": self.config["model"],
            "provider_call_attempted": provider_call_made,
            "provider_call_made": provider_call_made,
            "failed_at": _utc_now(),
            "error": error,
        }
        self._refresh_progress()
        self.save()

    def mark_status(self, status: str, reason: Optional[str] = None) -> None:
        self.payload["status"] = status
        if reason:
            self.payload["last_error"] = reason
        self._refresh_progress()
        self.save()


class CallBudget:
    def __init__(self, used: int, maximum: int):
        self.used = used
        self.maximum = maximum
        self._lock = threading.Lock()

    def ensure_available(self) -> None:
        with self._lock:
            if self.used >= self.maximum:
                raise CallBudgetExceeded(f"Provider call budget exhausted ({self.used}/{self.maximum})")

    def consume(self) -> None:
        with self._lock:
            if self.used >= self.maximum:
                raise CallBudgetExceeded(f"Provider call budget exhausted ({self.used}/{self.maximum})")
            self.used += 1


class BudgetedGenerator:
    """Adapter that reserves a call immediately before a planned generation."""

    def __init__(self, generator: AnswerGenerator, budget: CallBudget, call_by_question: Dict[str, bool]):
        self._generator = generator
        self._budget = budget
        self._call_by_question = call_by_question

    @property
    def provider(self):
        return self._generator.provider

    @property
    def retrieval_mode(self):
        return self._generator.retrieval_mode

    def generate(self, question: str, top_k: int = 5):
        if self._call_by_question.get(question, True):
            self._budget.consume()
        return self._generator.generate(question=question, top_k=top_k)


def _retrieval_plan(mode: str, top_k: int) -> Dict[str, bool]:
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.retriever import DenseRetriever

    retriever = HybridRetriever() if mode == "hybrid" else DenseRetriever()
    try:
        answerable, out_of_scope = load_questions()
        plan: Dict[str, bool] = {}
        for item in answerable + out_of_scope:
            retrieved = retriever.retrieve(item["question"], top_k=top_k)
            chunks = getattr(retrieved, "chunks", retrieved if isinstance(retrieved, list) else [])
            plan[item["id"]] = bool(chunks)
        return plan
    finally:
        if hasattr(retriever, "close"):
            retriever.close()


def preflight(mode: str, model: str, top_k: int, max_calls: int) -> Dict[str, Any]:
    config = build_run_config(mode, model, top_k, max_calls)
    provider = get_llm_provider(provider_type="groq", model_name=model)
    if provider.name != "groq" or provider.default_model != model:
        raise CheckpointError("Provider factory did not return the requested Groq provider/model")
    plan = _retrieval_plan(mode, top_k)
    expected = sum(1 for will_call in plan.values() if will_call)
    report = {
        "status": "preflight_only",
        "provider": "groq",
        "model": model,
        "retrieval_mode": mode,
        "question_count": len(plan),
        "retrieval_nonempty_questions": sum(plan.values()),
        "retrieval_empty_questions": len(plan) - sum(plan.values()),
        "warmup_calls": 0,
        "expected_provider_calls": expected,
        "max_call_budget": max_calls,
        "budget_sufficient_for_current_retrieval_plan": max_calls >= expected,
        "question_ids_skipping_generation": [qid for qid, will_call in plan.items() if not will_call],
        "configuration": config,
        "generation_calls_made": 0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_DIR / f"generation_evaluation_groq_{mode}_preflight.json", report)
    return report


def _temporary_question_file(directory: Path, name: str, items: Iterable[Dict[str, Any]]) -> Path:
    path = directory / name
    path.write_text(json.dumps(list(items), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _incomplete_artifacts(mode: str, checkpoint: CheckpointStore) -> None:
    payload = dict(checkpoint.payload)
    payload["status"] = checkpoint.payload.get("status", "incomplete")
    json_path = RESULTS_DIR / f"generation_evaluation_groq_{mode}_incomplete.json"
    md_path = RESULTS_DIR / f"generation_evaluation_groq_{mode}_incomplete.md"
    atomic_write_json(json_path, payload)
    counts = checkpoint.counts()
    md = (
        f"# Groq Generation Evaluation — {mode.title()} — Incomplete\n\n"
        f"- Provider/model: `groq` / `{checkpoint.config['model']}`\n"
        f"- Completed: {counts['completed_questions']} / {counts['total_questions']}\n"
        f"- Failed: {counts['failed_questions']}\n"
        f"- Remaining: {counts['remaining_questions']}\n"
        f"- Provider calls used: {counts['provider_calls_used']} / {checkpoint.config['max_call_budget']}\n"
        f"- Checkpoint: `{checkpoint.path}`\n"
    )
    md_path.write_text(md, encoding="utf-8")


def run_mode(mode: str, model: str, top_k: int, max_calls: int, resume: bool = False) -> Dict[str, Any]:
    config = build_run_config(mode, model, top_k, max_calls)
    checkpoint = CheckpointStore.open(checkpoint_path(mode), config, resume=resume)
    checkpoint.mark_status("running")
    answerable, out_of_scope = load_questions()
    items = [(item, "answerable") for item in answerable] + [(item, "out_of_scope") for item in out_of_scope]
    retrieval_plan = _retrieval_plan(mode, top_k)
    call_by_question = {item["question"]: retrieval_plan.get(item["id"], True) for item in answerable + out_of_scope}
    generator = AnswerGenerator(
        provider=(provider := get_llm_provider(provider_type="groq", model_name=model)),
        config=GenerationConfig(
            provider_type="groq",
            model_name=model,
            retrieval_mode=mode,
            top_k=top_k,
            max_completion_tokens=getattr(provider, "max_completion_tokens", None),
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
            # A provider call is attempted by the evaluator unless retrieval is empty;
            # budget is checked immediately before generation by the generator wrapper.
            try:
                with tempfile.TemporaryDirectory(prefix=f"groq_eval_{mode}_") as temp_dir:
                    temp = Path(temp_dir)
                    ans_path = _temporary_question_file(temp, "answerable.json", [item] if category == "answerable" else [])
                    oos_path = _temporary_question_file(temp, "oos.json", [item] if category == "out_of_scope" else [])
                    # The evaluator's retrieval result is authoritative.  It is run once
                    # for this question, with warmup disabled, so there are no hidden calls.
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
            except CallBudgetExceeded as exc:
                checkpoint.mark_status("budget_exhausted", str(exc))
                _incomplete_artifacts(mode, checkpoint)
                return checkpoint.payload
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
        final_json = RESULTS_DIR / f"generation_evaluation_groq_{mode}.json"
        final_md = RESULTS_DIR / f"generation_evaluation_groq_{mode}.md"
        save_generation_results(summary, completed_results, final_json, final_md)
        checkpoint.mark_status("complete")
        checkpoint.payload["final_artifacts"] = {"json": str(final_json), "markdown": str(final_md)}
        checkpoint.save()
        return checkpoint.payload
    finally:
        generator.close()

