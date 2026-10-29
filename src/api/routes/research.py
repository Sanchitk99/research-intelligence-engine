"""Research question answering and evidence synthesis endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    CitationResponse,
    RetrievedChunkResponse,
)
from src.api.dependencies import get_research_service
from src.ui.service import ResearchEngineService
from src.generation.models import GeneratedAnswer, AnswerStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["Research"])


@router.post("/ask", response_model=AskQuestionResponse)
def ask_research_question(
    req: AskQuestionRequest,
    service: ResearchEngineService = Depends(get_research_service),
) -> AskQuestionResponse:
    """Execute end-to-end evidence retrieval, synthesis, and page citation validation.

    Guarantees page-level provenance and explicit abstention on insufficient context.
    """
    clean_q = (req.question or "").strip()
    if not clean_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty or whitespace-only.",
        )

    try:
        ans: GeneratedAnswer = service.ask_question(
            question=clean_q,
            provider_name=req.provider,
            model_name=req.model,
            retrieval_mode=req.retrieval_mode,
            top_k=req.top_k,
            min_score=req.min_score,
            enable_intent_routing=req.enable_intent_routing,
            fallback_providers=req.fallback_providers,
        )
    except Exception as e:
        logger.error(f"Error during question answering: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the answer: {str(e)}",
        )

    # Map citations
    citations = [
        CitationResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            physical_page_number=c.physical_page_number,
            printed_page_number=c.printed_page_number,
            excerpt=c.excerpt,
        )
        for c in ans.citations
    ]

    # Map retrieved chunks
    retrieved_chunks = [
        RetrievedChunkResponse(
            chunk_id=rc.chunk_id,
            document_id=rc.document_id,
            document_title=rc.document_title,
            physical_page_number=rc.physical_page_number,
            printed_page_number=rc.printed_page_number,
            text=rc.text,
            score=round(rc.score, 4) if rc.score is not None else 0.0,
        )
        for rc in ans.retrieved_chunks
    ]

    # Convert status enum to string value
    status_str = ans.status.value if isinstance(ans.status, AnswerStatus) else str(ans.status)

    return AskQuestionResponse(
        question=ans.question,
        answer=ans.answer,
        status=status_str,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        model_name=ans.model_name,
        provider_name=ans.provider_name,
        retrieval_mode=ans.retrieval_mode,
        latency_ms=round(ans.latency_ms, 1),
        timing_breakdown=ans.timing_breakdown,
        confidence_notes=ans.confidence_notes,
        prompt_tokens=ans.prompt_tokens,
        completion_tokens=ans.completion_tokens,
        total_tokens=ans.total_tokens,
        failover_occurred=ans.failover_occurred,
        primary_provider=ans.primary_provider,
        fallback_attempts=ans.fallback_attempts,
    )
