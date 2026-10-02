"""The interface of a language model."""

from collections.abc import Mapping
from typing import Protocol

# A JSON Schema, as a plain mapping.
type Schema = Mapping[str, object]


class LanguageModelError(RuntimeError):
    """The model could not be reached or did not give a usable reply."""


class LanguageModel(Protocol):
    """Writes text in reply to a prompt.

    The rest of the system only knows this much about language models, so that
    a model running on this machine and one behind a remote API are
    interchangeable, and tests can script the replies.
    """

    @property
    def name(self) -> str:
        """Identifies the model."""
        ...

    def generate(self, prompt: str, *, system: str = "", schema: Schema | None = None) -> str:
        """Return the model's reply to a prompt.

        Args:
            prompt: What the model is asked.
            system: Standing instructions that frame the prompt.
            schema: If given, the reply is a JSON document that conforms to
                this JSON Schema.

        Raises:
            LanguageModelError: If no reply could be obtained.
        """
        ...
