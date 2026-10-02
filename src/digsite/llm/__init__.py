"""Language models: text in, text out."""

from digsite.llm.base import LanguageModel, LanguageModelError, Schema
from digsite.llm.ollama import DEFAULT_URL, OllamaModel

DEFAULT_MODEL = "gemma3:4b"

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "LanguageModel",
    "LanguageModelError",
    "OllamaModel",
    "Schema",
    "create_language_model",
]


def create_language_model(name: str = DEFAULT_MODEL, *, url: str = DEFAULT_URL) -> LanguageModel:
    """Build the language model with the given name.

    Every model is served by Ollama for now; this is the one place that would
    choose another client for another kind of name.
    """
    return OllamaModel(name, base_url=url)
