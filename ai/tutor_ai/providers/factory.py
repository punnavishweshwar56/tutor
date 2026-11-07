"""Factory for creating configured AI model providers."""

from ..settings import Settings
from .base import ModelProvider
from .gemini import GeminiProvider
from .mock import MockProvider
from .ollama import OllamaProvider


class ProviderNotConfigured(RuntimeError):
    """Raised when the requested provider is not implemented."""


def build_provider(settings: Settings) -> ModelProvider:
    """Build the model provider selected in application settings."""

    if settings.provider == "mock":
        return MockProvider(
            dimensions=settings.embedding_dimensions
        )

    if settings.provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_url,
            chat_model=settings.ollama_chat_model,
            embedding_model=settings.ollama_embed_model,
            dimensions=settings.embedding_dimensions,
            keep_alive=settings.ollama_keep_alive,
            timeout_seconds=settings.ollama_timeout_seconds,
            connect_timeout=settings.ollama_connect_timeout,
            max_batch_size=settings.ollama_max_batch_size,
        )

    if settings.provider == "gemini":
        if settings.gemini_api_key is None:
            raise ProviderNotConfigured(
                "TUTOR_GEMINI_API_KEY is required for Gemini."
            )

        ollama_provider = OllamaProvider(
            base_url=settings.ollama_url,
            chat_model=settings.ollama_chat_model,
            embedding_model=settings.ollama_embed_model,
            dimensions=settings.embedding_dimensions,
            keep_alive=settings.ollama_keep_alive,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            chat_model=settings.gemini_chat_model,
            ollama=ollama_provider,
            timeout_seconds=settings.turn_timeout_seconds,
        )

    raise ProviderNotConfigured(
        "The hosted provider is not implemented yet."
    )