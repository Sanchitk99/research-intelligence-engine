"""Factory function for initializing configured LLM providers."""

import os
from typing import Optional, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.generation.providers.base import BaseLLMProvider, ProviderError
from src.generation.providers.mock import MockLLMProvider
from src.generation.providers.openai_provider import OpenAIProvider
from src.generation.providers.gemini_provider import GeminiProvider
from src.generation.providers.groq_provider import GroqProvider


def get_llm_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Instantiate and return the configured LLM provider.

    Resolution order when provider_type is None or 'auto':
    1. Environment variable 'LLM_PROVIDER' (if set to 'openai', 'gemini', 'groq', 'mock', 'ollama')
    2. 'openai' if OPENAI_API_KEY is detected in environment
    3. 'gemini' if GEMINI_API_KEY is detected in environment
    4. 'groq' if GROQ_API_KEY is detected in environment
    5. 'mock' as safe fallback for local/offline testing

    Args:
        provider_type: Explicit provider name ('auto', 'openai', 'gemini', 'groq', 'mock', 'ollama').
        model_name: Optional default model override.
        **kwargs: Additional parameters forwarded to provider constructor.

    Returns:
        Instance of BaseLLMProvider.
    """
    chosen = (provider_type or os.environ.get("LLM_PROVIDER", "auto")).lower()

    if chosen == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            chosen = "openai"
        elif os.environ.get("GEMINI_API_KEY"):
            chosen = "gemini"
        elif os.environ.get("GROQ_API_KEY"):
            chosen = "groq"
        else:
            chosen = "mock"

    if chosen == "openai":
        model = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(default_model=model, **kwargs)

    elif chosen == "gemini":
        model = model_name or os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
        return GeminiProvider(default_model=model, **kwargs)

    elif chosen == "groq":
        model = model_name or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        return GroqProvider(default_model=model, **kwargs)

    elif chosen == "ollama":
        base_url = kwargs.pop("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        model = model_name or os.environ.get("OLLAMA_MODEL", "llama3")
        api_key = kwargs.pop("api_key", os.environ.get("OLLAMA_API_KEY", "ollama"))
        return OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            default_model=model,
            provider_name="ollama",
            **kwargs,
        )

    elif chosen == "mock":
        model = model_name or "mock-model"
        return MockLLMProvider(model_name=model, **kwargs)

    else:
        raise ProviderError(
            f"Unknown LLM provider '{chosen}'. Supported providers are: 'openai', 'gemini', 'groq', 'ollama', 'mock', 'auto'."
        )

