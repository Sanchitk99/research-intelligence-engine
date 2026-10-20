from unittest.mock import patch

import pytest

from src.evaluation.groq_generation_runner import CheckpointStore
from src.generation.providers.base import LLMResponse, RateLimitError
from src.generation.providers.groq_provider import GroqProvider
from src.generation.providers.openai_provider import OpenAIProvider


def test_rate_limit_header_parsing_and_token_classification():
    retry_after, reset_tokens, kind = OpenAIProvider._rate_limit_metadata(
        {
            "retry-after": "2.5s",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "1500ms",
        },
        '{"error":{"type":"tokens","message":"tokens per minute exceeded"}}',
    )
    assert retry_after == 2.5
    assert reset_tokens == 1.5
    assert kind == "tokens"


def test_rate_limit_delay_uses_largest_signal():
    error = RateLimitError("limited", retry_after_seconds=2.0, reset_tokens_seconds=4.0)
    assert error.retry_delay_seconds == 4.0


def test_groq_retries_bounded_with_delay_and_buffer():
    provider = GroqProvider(api_key="test", default_model="openai/gpt-oss-20b")
    provider._max_rate_limit_retries = 1
    provider._rate_limit_buffer_seconds = 0.25
    limited = RateLimitError("limited", retry_after_seconds=2.0, limit_kind="tokens")
    response = LLMResponse(content="{}", model="openai/gpt-oss-20b")
    with patch("src.generation.providers.openai_provider.OpenAIProvider.generate", side_effect=[limited, response]), patch(
        "src.generation.providers.groq_provider.time.sleep"
    ) as sleep:
        assert provider.generate(messages=[{"role": "user", "content": "hi"}]) == response
    sleep.assert_called_once_with(2.25)


def test_groq_stops_when_reset_information_is_unavailable():
    provider = GroqProvider(api_key="test")
    limited = RateLimitError("limited", limit_kind="tokens")
    with patch("src.generation.providers.openai_provider.OpenAIProvider.generate", side_effect=limited), patch(
        "src.generation.providers.groq_provider.time.sleep"
    ) as sleep:
        with pytest.raises(RateLimitError, match="did not include retry/reset timing"):
            provider.generate(messages=[{"role": "user", "content": "hi"}])
    sleep.assert_not_called()


def test_groq_retry_budget_is_bounded():
    provider = GroqProvider(api_key="test")
    provider._max_rate_limit_retries = 2
    limited = RateLimitError("limited", retry_after_seconds=0.0, limit_kind="tokens")
    with patch("src.generation.providers.openai_provider.OpenAIProvider.generate", side_effect=limited), patch(
        "src.generation.providers.groq_provider.time.sleep"
    ) as sleep:
        with pytest.raises(RateLimitError, match="retry budget exhausted"):
            provider.generate(messages=[{"role": "user", "content": "hi"}])
    assert sleep.call_count == 2


def test_checkpoint_preserves_completed_results_while_failed_entry_remains_retryable(tmp_path):
    config = {
        "provider": "groq", "model": "openai/gpt-oss-20b", "retrieval_mode": "dense", "top_k": 5,
        "max_call_budget": 28, "warmup_calls": 0, "question_ids": ["Q01", "Q02", "Q03"],
        "answerable_questions_sha256": "a", "out_of_scope_questions_sha256": "b",
        "prompt_sha256": "p", "scoring_sha256": "s", "scoring_version": "generation-evaluator-v1",
    }
    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(path, config)
    store.mark_complete("Q01", {"question_id": "Q01"}, provider_call_made=True)
    store.mark_complete("Q02", {"question_id": "Q02"}, provider_call_made=True)
    store.mark_failed("Q03", {"type": "RateLimitError", "message": "429"}, provider_call_made=True)
    resumed = CheckpointStore.open(path, config, resume=True)
    assert resumed.completed_ids() == {"Q01", "Q02"}
    assert resumed.payload["questions"]["Q01"]["result"] == {"question_id": "Q01"}
    assert resumed.payload["questions"]["Q03"]["state"] == "failed"
