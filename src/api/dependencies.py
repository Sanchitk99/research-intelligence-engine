"""Dependency injection and service lifecycle management for FastAPI."""

import logging
from pathlib import Path
from typing import Optional

from src.ui.service import ResearchEngineService

logger = logging.getLogger(__name__)

# Global singleton instance of the core engine service
_service_instance: Optional[ResearchEngineService] = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return project root directory."""
    return PROJECT_ROOT


def initialize_service(
    qdrant_path: Optional[Path] = None,
    collection_name: str = "research_corpus",
    chunks_file: Optional[Path] = None,
) -> ResearchEngineService:
    """Initialize the singleton engine service and warm up stores."""
    global _service_instance
    if _service_instance is not None:
        return _service_instance

    resolved_qdrant = qdrant_path or (PROJECT_ROOT / "data" / "qdrant")
    resolved_chunks = chunks_file or (PROJECT_ROOT / "data" / "chunked" / "corpus_chunks.json")

    logger.info(f"Initializing ResearchEngineService (qdrant: {resolved_qdrant}, chunks: {resolved_chunks})")
    _service_instance = ResearchEngineService(
        qdrant_path=resolved_qdrant,
        collection_name=collection_name,
        chunks_file=resolved_chunks,
    )
    return _service_instance


def get_research_service() -> ResearchEngineService:
    """FastAPI dependency to retrieve the active engine service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = initialize_service()
    return _service_instance


def shutdown_service() -> None:
    """Safely close engine service and release Qdrant storage locks."""
    global _service_instance
    if _service_instance is not None:
        logger.info("Closing ResearchEngineService...")
        try:
            _service_instance.close()
        except Exception as e:
            logger.warning(f"Error closing service instance: {e}")
        finally:
            _service_instance = None
