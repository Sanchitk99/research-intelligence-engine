import json
from unittest.mock import MagicMock, patch

import httpx

from src.generation.generator import AnswerGenerator
from src.generation.models import GenerationConfig
from src.generation.providers.base import LLMResponse
from src.generation.providers.base import ProviderError
from src.generation.providers.groq_provider import GROQ_GENERATION_JSON_SCHEMA, groq_generation_response_format
from src.generation.providers.groq_provider import GroqProvider
from src.retrieval.models import RetrievedChunk, RetrievalResult


class CapturingProvider:
    name = "groq"
    default_model = "openai/gpt-oss-20b"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(
            content=json.dumps(
                {
                    "status": "answered",
                    "answer": "BART is the generator.",
                    "citations": [{"chunk_id": "P1_p01_c01", "excerpt": "BART is the generator."}],
                    "confidence_notes": None,
                }
            ),
            model=self.default_model,
        )


class StubRetriever:
    def retrieve(self, query, top_k=5, min_score=None):
        chunk = RetrievedChunk(
            chunk_id="P1_p01_c01",
            document_id="P1",
            document_title="Paper",
            physical_page_index=0,
            physical_page_number=1,
            chunk_index_in_page=0,
            chunk_index_in_doc=0,
            start_char_idx=0,
            end_char_idx=22,
            char_count=22,
            word_count=4,
            text="BART is the generator.",
            score=1.0,
        )
        return RetrievalResult(
            query=query,
            chunks=[chunk],
            total_retrieved=1,
            collection_name="test",
            execution_time_ms=0.0,
        )


def test_groq_strict_schema_has_required_closed_objects():
    response_format = groq_generation_response_format()
    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    citation_schema = schema["properties"]["citations"]["items"]
    assert citation_schema["required"] == ["chunk_id", "excerpt"]
    assert citation_schema["additionalProperties"] is False
    assert schema["properties"]["confidence_notes"]["type"] == ["string", "null"]
    assert schema == GROQ_GENERATION_JSON_SCHEMA


def test_answer_generator_requests_groq_strict_schema_and_keeps_application_validation():
    provider = CapturingProvider()
    generator = AnswerGenerator(
        retriever=StubRetriever(),
        provider=provider,
        config=GenerationConfig(provider_type="groq", model_name=provider.default_model),
        retrieval_mode="dense",
    )
    result = generator.generate("What generator model does RAG use?")

    # AnswerGenerator keeps the provider-neutral interface. GroqProvider
    # transforms this value into the strict schema at the HTTP boundary.
    assert provider.calls[0]["response_format"] == "json_object"
    assert result.status.value == "answered"
    assert result.citations[0].chunk_id == "P1_p01_c01"


def test_groq_sends_configured_max_completion_tokens_without_legacy_max_tokens(monkeypatch):
    monkeypatch.setenv("GROQ_MAX_COMPLETION_TOKENS", "3072")
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    provider = GroqProvider(api_key="test", default_model="openai/gpt-oss-20b")
    with patch("httpx.Client.post", return_value=response) as post:
        provider.generate(messages=[{"role": "user", "content": "hi"}], response_format="json_object")
    payload = post.call_args.kwargs["json"]
    assert payload["max_completion_tokens"] == 3072
    assert "max_tokens" not in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_groq_http_400_is_not_retried():
    response = MagicMock(spec=httpx.Response)
    response.status_code = 400
    response.text = '{"error":{"code":"json_validate_failed"}}'
    provider = GroqProvider(api_key="test")
    with patch("httpx.Client.post", return_value=response) as post:
        try:
            provider.generate(messages=[{"role": "user", "content": "hi"}], response_format="json_object")
        except ProviderError:
            pass
    assert post.call_count == 1
