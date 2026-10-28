"""Deterministic unit and integration tests for automatic LLM provider failover.

Covers:
1. Primary provider succeeds without fallback.
2. Gemini returns 503 and the next configured provider succeeds.
3. Gemini returns 429 and the next configured provider succeeds.
4. Primary authentication/configuration failure skips to a valid configured fallback where appropriate.
5. All providers fail and the application returns a clear generation error.
6. Missing fallback credentials are handled safely.
7. Retrieval occurs only once during failover.
8. Final citations are validated normally after fallback generation.
9. The response reports the actual successful provider/model.
10. Legitimate abstention is not mistaken for provider failure.
11. Bad requests (HTTP 400) are not blindly retried across providers.
12. Secrets, authorization headers, and API keys are redacted from diagnostics.
13. Fallback chain resolution through settings and environment variables.
"""

import json
import os
from typing import List
from unittest.mock import MagicMock, patch
import pytest

from src.retrieval.models import RetrievedChunk, RetrievalResult
from src.generation.models import (
    AnswerStatus,
    Citation,
    GenerationConfig,
    GeneratedAnswer,
)
from src.generation.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ProviderAuthError,
    ProviderTimeoutError,
    RateLimitError,
)
from src.generation.providers.mock import MockLLMProvider
from src.generation.providers.gemini_provider import GeminiProvider
from src.generation.generator import AnswerGenerator
from src.generation.failover import (
    classify_provider_error,
    sanitize_error_message,
    resolve_fallback_chain,
    is_provider_configured,
    get_default_fallback_providers,
)


@pytest.fixture
def sample_retrieved_chunks() -> List[RetrievedChunk]:
    """Sample chunks for deterministic generation tests."""
    return [
        RetrievedChunk(
            chunk_id="P1_p03_c04",
            document_id="P1",
            document_title="Attention Is All You Need",
            physical_page_index=2,
            physical_page_number=3,
            printed_page_number=3,
            chunk_index_in_page=3,
            chunk_index_in_doc=12,
            start_char_idx=1000,
            end_char_idx=1800,
            char_count=800,
            word_count=120,
            text="The multi-head attention mechanism consists of several attention layers.",
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="P2_p04_c02",
            document_id="P2",
            document_title="BERT: Pre-training of Deep Bidirectional Transformers",
            physical_page_index=3,
            physical_page_number=4,
            printed_page_number=None,
            chunk_index_in_page=1,
            chunk_index_in_doc=8,
            start_char_idx=500,
            end_char_idx=1200,
            char_count=700,
            word_count=110,
            text="BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
            score=0.85,
        ),
    ]


@pytest.fixture
def sample_retrieval_result(sample_retrieved_chunks) -> RetrievalResult:
    return RetrievalResult(
        query="What is multi-head attention?",
        chunks=sample_retrieved_chunks,
        total_retrieved=len(sample_retrieved_chunks),
        collection_name="research_corpus",
        retrieval_mode="dense",
    )


def test_1_primary_provider_succeeds_without_fallback(sample_retrieval_result):
    """1. Primary provider succeeds immediately; fallback provider is never called."""
    primary = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "Multi-head attention consists of multiple parallel attention layers.",
            "citations": [
                {
                    "chunk_id": "P1_p03_c04",
                    "excerpt": "The multi-head attention mechanism consists of several attention layers.",
                }
            ],
            "confidence_notes": "Clear evidence in P1.",
        },
        model_name="primary-mock",
    )
    fallback = MockLLMProvider(
        default_response="Should not be called",
        model_name="fallback-mock",
    )

    generator = AnswerGenerator(
        provider=primary,
        fallback_providers=[fallback],
        config=GenerationConfig(max_provider_retries=0),
    )

    result: GeneratedAnswer = generator.generate(
        question="What is multi-head attention?",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ANSWERED
    assert result.provider_name == "mock"
    assert result.model_name == "primary-mock"
    assert result.failover_occurred is False
    assert len(primary.call_history) == 1
    assert len(fallback.call_history) == 0


def test_2_gemini_503_fails_over_to_next_configured_provider(sample_retrieval_result):
    """2. Gemini returns HTTP 503 and the next configured provider succeeds."""
    primary_gemini = MockLLMProvider(
        simulate_exception=ProviderError("Gemini API returned error HTTP 503: The model is overloaded.", status_code=503),
        model_name="gemini-flash-lite-latest",
    )
    # Give provider name 'gemini' for realistic test
    primary_gemini._name = "gemini"
    type(primary_gemini).name = property(lambda self: getattr(self, "_name", "mock"))

    fallback_provider = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "Multi-head attention runs several attention mechanisms in parallel.",
            "citations": [
                {
                    "chunk_id": "P1_p03_c04",
                    "excerpt": "The multi-head attention mechanism consists of several attention layers.",
                }
            ],
            "confidence_notes": "Sourced from P1.",
        },
        model_name="fallback-model",
    )
    fallback_provider._name = "groq"

    generator = AnswerGenerator(
        provider=primary_gemini,
        fallback_providers=[fallback_provider],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(
        question="What is multi-head attention?",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ANSWERED
    assert result.provider_name == "groq"
    assert result.model_name == "fallback-model"
    assert result.failover_occurred is True
    assert result.primary_provider == "gemini"
    assert "[Failover from gemini to groq]" in result.confidence_notes
    assert len(fallback_provider.call_history) == 1


def test_3_gemini_429_fails_over_to_next_configured_provider(sample_retrieval_result):
    """3. Gemini returns HTTP 429 (rate limit) and the next configured provider succeeds."""
    primary_gemini = MockLLMProvider(
        simulate_exception=RateLimitError("Gemini rate limit exceeded: quota exhausted", retry_after_seconds=60.0),
        model_name="gemini-flash-lite-latest",
    )
    primary_gemini._name = "gemini"
    type(primary_gemini).name = property(lambda self: getattr(self, "_name", "mock"))

    fallback_provider = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "BERT is designed to pre-train deep bidirectional representations.",
            "citations": [
                {
                    "chunk_id": "P2_p04_c02",
                    "excerpt": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
                }
            ],
            "confidence_notes": "Sourced from P2.",
        },
        model_name="openai-fallback",
    )
    fallback_provider._name = "openai"

    generator = AnswerGenerator(
        provider=primary_gemini,
        fallback_providers=[fallback_provider],
        config=GenerationConfig(max_provider_retries=0, max_retry_delay_seconds=2.0),
    )

    result = generator.generate(
        question="What is BERT?",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ANSWERED
    assert result.provider_name == "openai"
    assert result.model_name == "openai-fallback"
    assert result.failover_occurred is True
    assert result.primary_provider == "gemini"
    assert len(fallback_provider.call_history) == 1


def test_4_primary_auth_failure_skips_immediately_to_fallback(sample_retrieval_result):
    """4. Primary authentication/configuration failure skips to a valid fallback without retry loops."""
    primary_bad_auth = MockLLMProvider(
        simulate_exception=ProviderAuthError("Gemini API key is invalid or rejected.", status_code=401),
        model_name="gemini-flash-lite-latest",
    )
    primary_bad_auth._name = "gemini"
    type(primary_bad_auth).name = property(lambda self: getattr(self, "_name", "mock"))

    fallback = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "Attention layers allow multi-head representation learning.",
            "citations": [
                {
                    "chunk_id": "P1_p03_c04",
                    "excerpt": "The multi-head attention mechanism consists of several attention layers.",
                }
            ],
            "confidence_notes": None,
        },
        model_name="safe-fallback",
    )
    fallback._name = "mock"

    generator = AnswerGenerator(
        provider=primary_bad_auth,
        fallback_providers=[fallback],
        config=GenerationConfig(max_provider_retries=2),  # Should NOT retry auth error
    )

    result = generator.generate(
        question="Explain multi-head attention.",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ANSWERED
    assert result.provider_name == "mock"
    assert result.model_name == "safe-fallback"
    assert result.failover_occurred is True
    # Verify primary was called only ONCE despite max_provider_retries=2
    assert len(primary_bad_auth.call_history) == 1
    assert len(fallback.call_history) == 1


def test_5_all_providers_fail_returns_clear_generation_error(sample_retrieval_result):
    """5. All providers fail and the application returns a clear generation error (not abstention)."""
    p1 = MockLLMProvider(
        simulate_exception=ProviderError("Gemini HTTP 503 unavailable", status_code=503),
        model_name="gemini-model",
    )
    p2 = MockLLMProvider(
        simulate_exception=RateLimitError("Groq HTTP 429 rate limit"),
        model_name="groq-model",
    )
    p3 = MockLLMProvider(
        simulate_exception=ProviderTimeoutError("OpenAI timeout after 30s"),
        model_name="openai-model",
    )

    generator = AnswerGenerator(
        provider=p1,
        fallback_providers=[p2, p3],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(
        question="What is multi-head attention?",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ERROR
    assert "All configured LLM providers failed to generate an answer" in result.answer
    assert "Failover chain exhausted" in result.confidence_notes
    assert result.failover_occurred is True
    assert len(result.citations) == 0


def test_6_missing_fallback_credentials_handled_safely(sample_retrieval_result):
    """6. Missing fallback credentials are skipped safely and subsequent fallback succeeds."""
    primary = MockLLMProvider(
        simulate_exception=ProviderError("Primary HTTP 503", status_code=503),
        model_name="p1",
    )
    fallback_no_creds = MockLLMProvider(
        simulate_exception=ProviderAuthError("Groq API key is missing. Set the GROQ_API_KEY env var."),
        model_name="p2-groq",
    )
    fallback_valid = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "BERT uses bidirectional transformer encoders.",
            "citations": [
                {
                    "chunk_id": "P2_p04_c02",
                    "excerpt": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
                }
            ],
            "confidence_notes": "Verified from P2.",
        },
        model_name="p3-working",
    )

    generator = AnswerGenerator(
        provider=primary,
        fallback_providers=[fallback_no_creds, fallback_valid],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(
        question="Describe BERT.",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ANSWERED
    assert result.model_name == "p3-working"
    assert result.failover_occurred is True


def test_7_retrieval_occurs_only_once_during_failover(sample_retrieved_chunks):
    """7. Retrieval executes exactly once even when primary fails and fallback succeeds."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="What is attention?",
        chunks=sample_retrieved_chunks,
        total_retrieved=len(sample_retrieved_chunks),
        collection_name="research_corpus",
        retrieval_mode="dense",
    )

    p1 = MockLLMProvider(simulate_exception=ProviderError("503 Server Error", status_code=503))
    p2 = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "Attention mechanism details.",
            "citations": [
                {
                    "chunk_id": "P1_p03_c04",
                    "excerpt": "The multi-head attention mechanism consists of several attention layers.",
                }
            ],
            "confidence_notes": None,
        }
    )

    generator = AnswerGenerator(
        retriever=mock_retriever,
        provider=p1,
        fallback_providers=[p2],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(question="What is attention?")

    # Verify retriever.retrieve was called strictly ONCE
    assert mock_retriever.retrieve.call_count == 1
    assert result.status == AnswerStatus.ANSWERED


def test_8_final_citations_validated_normally_after_fallback(sample_retrieval_result):
    """8. Citations from fallback provider undergo complete excerpt grounding and normalization validation."""
    primary = MockLLMProvider(simulate_exception=RateLimitError("Rate limit hit"))
    fallback = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "Multi-head attention combines several layers.",
            "citations": [
                {
                    "chunk_id": "P1_p03_c04",
                    "excerpt": "The multi-head attention mechanism consists of several attention layers.",
                },
                {
                    "chunk_id": "P99_hallucinated_chunk",
                    "excerpt": "Hallucinated excerpt not in retrieved context.",
                },
            ],
            "confidence_notes": "Initial notes.",
        }
    )

    generator = AnswerGenerator(
        provider=primary,
        fallback_providers=[fallback],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(
        question="What is multi-head attention?",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ANSWERED
    # Valid citation is kept
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "P1_p03_c04"
    # Hallucinated citation is rejected and flagged in notes
    assert "Rejected hallucinated chunk_id 'P99_hallucinated_chunk'" in (result.confidence_notes or "")


def test_9_response_reports_actual_successful_provider_and_model(sample_retrieval_result):
    """9. The response accurately reflects the provider and model that actually completed generation."""
    primary = MockLLMProvider(
        simulate_exception=ProviderError("502 Bad Gateway", status_code=502),
        model_name="primary-gemini-v1",
    )
    primary._name = "gemini"
    type(primary).name = property(lambda self: getattr(self, "_name", "mock"))

    fallback = MockLLMProvider(
        default_response={
            "status": "answered",
            "answer": "Answer from Groq model.",
            "citations": [
                {
                    "chunk_id": "P1_p03_c04",
                    "excerpt": "The multi-head attention mechanism consists of several attention layers.",
                }
            ],
            "confidence_notes": None,
        },
        model_name="llama-3.3-70b-versatile",
    )
    fallback._name = "groq"
    type(fallback).name = property(lambda self: getattr(self, "_name", "mock"))

    generator = AnswerGenerator(
        provider=primary,
        fallback_providers=[fallback],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(
        question="Explain attention.",
        retrieval_result=sample_retrieval_result,
    )

    assert result.provider_name == "groq"
    assert result.model_name == "llama-3.3-70b-versatile"
    assert result.primary_provider == "gemini"
    assert result.failover_occurred is True


def test_10_legitimate_abstention_not_mistaken_for_provider_failure(sample_retrieval_result):
    """10. Legitimate model abstentions (e.g. insufficient context) are preserved and do not trigger failover."""
    primary = MockLLMProvider(
        default_response={
            "status": "abstained_insufficient_evidence",
            "answer": "The provided research passages do not contain information regarding quantum computing.",
            "citations": [],
            "confidence_notes": "Quantum computing is unmentioned in P1 and P2.",
        },
        model_name="primary-abstainer",
    )
    fallback = MockLLMProvider(
        default_response="Should not be called during legitimate abstention",
        model_name="fallback-unused",
    )

    generator = AnswerGenerator(
        provider=primary,
        fallback_providers=[fallback],
        config=GenerationConfig(max_provider_retries=0),
    )

    result = generator.generate(
        question="How does quantum computing affect multi-head attention?",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
    assert result.model_name == "primary-abstainer"
    assert result.failover_occurred is False
    assert len(primary.call_history) == 1
    assert len(fallback.call_history) == 0


def test_11_bad_request_not_blindly_retried(sample_retrieval_result):
    """11. HTTP 400 Bad Request indicates an invalid payload and must not be retried across providers."""
    primary = MockLLMProvider(
        simulate_exception=ProviderError("Gemini API returned error HTTP 400: Invalid JSON payload argument", status_code=400),
    )
    fallback = MockLLMProvider(
        default_response="Should not be called on 400",
    )

    generator = AnswerGenerator(
        provider=primary,
        fallback_providers=[fallback],
        config=GenerationConfig(max_provider_retries=2),
    )

    result = generator.generate(
        question="Malformed query test",
        retrieval_result=sample_retrieval_result,
    )

    assert result.status == AnswerStatus.ERROR
    assert "Invalid generation request rejected" in result.answer
    assert len(fallback.call_history) == 0


def test_12_secret_sanitization_in_diagnostics():
    """12. Secrets, API keys, and authorization headers are sanitized from diagnostics."""
    raw_error = (
        "Error calling https://api.groq.com/openai/v1?key=AIzaSySecretKey123456789: "
        "Authorization: Bearer sk-ant-api03-verysecrettoken9999999 rejected with x-goog-api-key: AIzaSyTopSecretKey"
    )
    sanitized = sanitize_error_message(raw_error)

    assert "AIzaSySecretKey123456789" not in sanitized
    assert "sk-ant-api03-verysecrettoken9999999" not in sanitized
    assert "AIzaSyTopSecretKey" not in sanitized
    assert "[REDACTED]" in sanitized


def test_13_fallback_chain_resolution_from_env():
    """13. Fallback chain parses comma-separated LLM_FALLBACK_PROVIDERS from environment."""
    primary = MockLLMProvider(model_name="primary")

    with patch.dict(os.environ, {"LLM_FALLBACK_PROVIDERS": "mock:custom-fallback"}):
        chain = resolve_fallback_chain(primary_provider=primary)
        assert len(chain) == 1
        assert chain[0].name == "mock"
        assert chain[0].default_model == "custom-fallback"
