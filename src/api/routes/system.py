"""System health and status endpoints."""

from fastapi import APIRouter, Depends
from src.api.schemas import HealthResponse, SystemStatusResponse, ProviderStatus, PaperSummaryResponse
from src.api.dependencies import get_research_service
from src.ui.service import ResearchEngineService

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Lightweight server health probe."""
    return HealthResponse(status="ok", service="Research Intelligence Engine API", version="1.0.0")


@router.get("/status", response_model=SystemStatusResponse)
def get_status(service: ResearchEngineService = Depends(get_research_service)) -> SystemStatusResponse:
    """Query current vector collection stats, paper counts, and sanitized provider availability."""
    raw_status = service.get_system_status()
    papers = [
        PaperSummaryResponse(
            document_id=p.document_id,
            title=p.title,
            total_pages=p.total_pages,
            total_chunks=p.total_chunks,
        )
        for p in raw_status.get("papers", [])
    ]

    providers = {
        name: ProviderStatus(
            configured=info.get("configured", False),
            default_model=info.get("default_model", ""),
        )
        for name, info in raw_status.get("providers", {}).items()
    }

    return SystemStatusResponse(
        collection_name=raw_status.get("collection_name", "research_corpus"),
        total_points=raw_status.get("total_points", 0),
        indexed_papers_count=raw_status.get("indexed_papers_count", 0),
        papers=papers,
        providers=providers,
        fallback_chain=raw_status.get("fallback_chain"),
    )
