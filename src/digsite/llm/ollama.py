"""Language models served by Ollama, on this machine or another."""

import logging

import httpx

from digsite.llm.base import LanguageModelError, Schema

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"
DEFAULT_CONTEXT_TOKENS = 8192

# Generating on a CPU takes minutes; finding out that nothing is listening should not.
_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


class OllamaModel:
    """A model behind Ollama's chat endpoint.

    Args:
        model: Name of the model, as `ollama list` shows it.
        base_url: Where Ollama listens.
        context_tokens: Size of the model's context window. Ollama's default
            is small and it cuts longer prompts without saying so, so the size
            is always sent.
        temperature: Randomness of the reply. 0 makes the most likely reply,
            which suits answers that must stick to their sources.
        seed: Seed of the sampler, so that a reply can be reproduced.
        client: HTTP client to use. One is created if none is given.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_URL,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        temperature: float = 0.0,
        seed: int = 0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._context_tokens = context_tokens
        self._options = {"temperature": temperature, "seed": seed, "num_ctx": context_tokens}
        self._client = client or httpx.Client(timeout=_TIMEOUT)

    @property
    def name(self) -> str:
        return self._model

    def generate(self, prompt: str, *, system: str = "", schema: Schema | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        request: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": self._options,
        }
        if schema is not None:
            request["format"] = dict(schema)

        try:
            response = self._client.post(self._url, json=request)
        except httpx.ConnectError as error:
            raise LanguageModelError(
                f"cannot reach Ollama at {self._url}; is it running? ({error})"
            ) from error
        except httpx.HTTPError as error:
            raise LanguageModelError(f"the request to Ollama failed: {error!r}") from error
        if response.status_code == httpx.codes.NOT_FOUND:
            raise LanguageModelError(
                f"Ollama does not have the model {self._model!r}; "
                f"run 'ollama pull {self._model}' first"
            )
        if response.is_error:
            raise LanguageModelError(
                f"Ollama answered {response.status_code}: {_error_message(response)}"
            )

        try:
            reply = response.json()
            content = reply["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise LanguageModelError("Ollama's reply was not in the expected format") from error
        if not isinstance(content, str):
            raise LanguageModelError("Ollama's reply was not in the expected format")
        self._check_limits(reply)
        return content

    def _check_limits(self, reply: dict[str, object]) -> None:
        """Warn when the prompt or the reply ran into the context window."""
        prompt_tokens = reply.get("prompt_eval_count")
        reply_tokens = reply.get("eval_count")
        logger.debug(
            "%s: %s prompt tokens, %s reply tokens", self._model, prompt_tokens, reply_tokens
        )
        if isinstance(prompt_tokens, int) and prompt_tokens >= self._context_tokens:
            logger.warning(
                "the prompt filled the context window of %d tokens and was cut",
                self._context_tokens,
            )
        if reply.get("done_reason") == "length":
            logger.warning("the reply was cut: the model ran out of context window")


def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]
    except ValueError, KeyError, TypeError:
        return response.text.strip() or "no details"
    return str(message)
