"""Optional CPU-only cross-encoder reranking for isolated experiments.

This module is intentionally not wired into the default dense or hybrid
retrievers.  It preserves the existing RetrievedChunk instances and only
changes their ordering for an opt-in experiment.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.retrieval.models import RetrievedChunk


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L2-v2"
MIN_AVAILABLE_RAM_MB = 3072


class RerankerError(RuntimeError):
    """Base error for the optional reranker experiment."""


class RerankerModelNotCachedError(RerankerError):
    """Raised when the requested model is not already available locally."""


class RerankerResourceError(RerankerError):
    """Raised when local resource safeguards reject an experiment."""


@dataclass(frozen=True)
class RerankResult:
    """Reranking output and measurements for one query."""

    chunks: List[RetrievedChunk]
    scores: List[float]
    candidate_count: int
    scoring_latency_ms: float


def model_cache_directory(model_name: str = DEFAULT_RERANKER_MODEL, cache_root: Optional[Path] = None) -> Path:
    """Return the Hugging Face hub cache directory for a model."""

    root = cache_root
    if root is None:
        configured = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if configured:
            root = Path(configured)
        else:
            hf_home = os.environ.get("HF_HOME")
            root = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    safe_name = model_name.replace("/", "--")
    return root / f"models--{safe_name}"


def is_model_cached(model_name: str = DEFAULT_RERANKER_MODEL, cache_root: Optional[Path] = None) -> bool:
    """Check for a usable local snapshot without loading model weights."""

    directory = model_cache_directory(model_name, cache_root)
    snapshots = directory / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            continue
        if any((snapshot / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")):
            return True
    return False


def available_ram_mb() -> Optional[float]:
    """Return available physical RAM, without requiring a new dependency."""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024 * 1024)
        return None

    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None


def require_resources(min_ram_mb: int = MIN_AVAILABLE_RAM_MB) -> float:
    """Enforce the minimum available-RAM guard."""

    available = available_ram_mb()
    if available is None:
        raise RerankerResourceError("Unable to determine available system RAM")
    if available < min_ram_mb:
        raise RerankerResourceError(
            f"Available RAM {available:.0f} MB is below the {min_ram_mb} MB experiment minimum"
        )
    return available


def active_ollama_models(base_url: str = "http://localhost:11434", timeout_seconds: float = 2.0) -> Optional[List[str]]:
    """Read Ollama's loaded-model list without issuing a generation request.

    ``None`` means the local endpoint was unavailable or could not be inspected.
    """

    request = Request(base_url.rstrip("/") + "/api/ps", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
            return [str(model.get("name")) for model in payload.get("models", []) if model.get("name")]
    except (OSError, URLError, ValueError, TimeoutError):
        return None


def _deduplicate_candidates(candidates: Iterable[RetrievedChunk]) -> List[RetrievedChunk]:
    """Deduplicate by stable chunk ID while preserving candidate order."""

    result: List[RetrievedChunk] = []
    seen = set()
    for chunk in candidates:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            result.append(chunk)
    return result


class CpuCrossEncoderReranker:
    """CPU-only CrossEncoder wrapper for the isolated reranking experiment."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = 4,
        max_candidates: int = 100,
        cache_root: Optional[Path] = None,
        cross_encoder_factory: Any = None,
    ) -> None:
        if batch_size < 1 or max_candidates < 1:
            raise ValueError("batch_size and max_candidates must be positive")
        if not is_model_cached(model_name, cache_root):
            raise RerankerModelNotCachedError(
                f"Model '{model_name}' is not available in the local Hugging Face cache; download it before this experiment"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        if cross_encoder_factory is None:
            from sentence_transformers import CrossEncoder

            cross_encoder_factory = CrossEncoder
        self.model = cross_encoder_factory(model_name, device="cpu")
        self.device = "cpu"

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int = 5) -> RerankResult:
        """Score and order candidates, retaining original chunk objects."""

        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        unique = _deduplicate_candidates(candidates)[: self.max_candidates]
        pairs = [(query, chunk.text) for chunk in unique]
        started = time.perf_counter()
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(unique):
            raise RerankerError(f"CrossEncoder returned {len(scores)} scores for {len(unique)} candidates")
        indexed = list(enumerate(zip(unique, scores)))
        indexed.sort(key=lambda item: (-item[1][1], item[0], item[1][0].chunk_id))
        selected = indexed[:top_k]
        return RerankResult(
            chunks=[item[1][0] for item in selected],
            scores=[item[1][1] for item in selected],
            candidate_count=len(unique),
            scoring_latency_ms=round(elapsed_ms, 2),
        )

