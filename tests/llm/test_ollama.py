import json
from collections.abc import Callable

import httpx
import pytest

from digsite.llm import LanguageModelError, OllamaModel

Handler = Callable[[httpx.Request], httpx.Response]


def reply(content: str = "Hello.", **extra: object) -> httpx.Response:
    body = {"model": "test", "message": {"role": "assistant", "content": content}, "done": True}
    return httpx.Response(200, json={**body, **extra})


def model(handler: Handler, **settings: object) -> OllamaModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaModel("test-model", client=client, **settings)  # type: ignore[arg-type]


class Recorder:
    """Answers every request with the same reply and keeps the requests."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response or reply()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._response

    @property
    def sent(self) -> dict[str, object]:
        body: dict[str, object] = json.loads(self.requests[-1].content)
        return body


def test_returns_the_text_of_the_reply() -> None:
    assert model(lambda request: reply("Forty-two.")).generate("What is six times seven?") == (
        "Forty-two."
    )


def test_asks_the_chat_endpoint_for_one_whole_reply() -> None:
    recorder = Recorder()

    model(recorder).generate("Hi")

    assert str(recorder.requests[0].url) == "http://localhost:11434/api/chat"
    assert recorder.sent["model"] == "test-model"
    assert recorder.sent["stream"] is False
    assert recorder.sent["messages"] == [{"role": "user", "content": "Hi"}]
    assert "format" not in recorder.sent


def test_system_instructions_go_before_the_prompt() -> None:
    recorder = Recorder()

    model(recorder).generate("Hi", system="Answer in Spanish.")

    assert recorder.sent["messages"] == [
        {"role": "system", "content": "Answer in Spanish."},
        {"role": "user", "content": "Hi"},
    ]


def test_a_schema_is_sent_as_the_format_of_the_reply() -> None:
    recorder = Recorder(reply('{"ok": true}'))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}

    text = model(recorder).generate("Is it ok?", schema=schema)

    assert recorder.sent["format"] == schema
    assert json.loads(text) == {"ok": True}


def test_the_context_size_is_always_sent() -> None:
    # Ollama's own default is small, and it cuts longer prompts without saying so.
    recorder = Recorder()

    model(recorder).generate("Hi")
    assert recorder.sent["options"] == {"temperature": 0.0, "seed": 0, "num_ctx": 8192}

    model(recorder, context_tokens=4096, temperature=0.7, seed=3).generate("Hi")
    assert recorder.sent["options"] == {"temperature": 0.7, "seed": 3, "num_ctx": 4096}


def test_the_address_of_the_server_can_be_changed() -> None:
    recorder = Recorder()

    model(recorder, base_url="http://gpu-box:11434/").generate("Hi")

    assert str(recorder.requests[0].url) == "http://gpu-box:11434/api/chat"


def test_reports_its_name() -> None:
    assert model(Recorder()).name == "test-model"


def test_a_server_that_is_not_running_is_explained() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LanguageModelError, match=r"cannot reach Ollama at .*is it running\?"):
        model(refuse).generate("Hi")


def test_a_timeout_is_reported() -> None:
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(LanguageModelError, match="request to Ollama failed"):
        model(stall).generate("Hi")


def test_a_model_that_was_not_pulled_is_explained() -> None:
    missing = httpx.Response(404, json={"error": "model 'test-model' not found"})

    with pytest.raises(LanguageModelError, match="run 'ollama pull test-model' first"):
        model(lambda request: missing).generate("Hi")


def test_other_errors_carry_the_message_of_the_server() -> None:
    broken = httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(LanguageModelError, match="answered 500: out of memory"):
        model(lambda request: broken).generate("Hi")


def test_an_error_without_a_json_body_carries_its_text() -> None:
    broken = httpx.Response(502, text="Bad Gateway")

    with pytest.raises(LanguageModelError, match="answered 502: Bad Gateway"):
        model(lambda request: broken).generate("Hi")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"done": True}),
        httpx.Response(200, json={"message": {"content": None}}),
        httpx.Response(200, json=["unexpected"]),
    ],
)
def test_a_reply_in_another_format_is_an_error(response: httpx.Response) -> None:
    with pytest.raises(LanguageModelError, match="not in the expected format"):
        model(lambda request: response).generate("Hi")


def test_warns_when_the_prompt_filled_the_context_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    full = reply("Hello.", prompt_eval_count=4096, eval_count=5)

    model(lambda request: full, context_tokens=4096).generate("a very long prompt")

    assert "filled the context window of 4096 tokens and was cut" in caplog.text


def test_warns_when_the_reply_was_cut(caplog: pytest.LogCaptureFixture) -> None:
    cut = reply("An unfinished", prompt_eval_count=100, eval_count=50, done_reason="length")

    assert model(lambda request: cut).generate("Hi") == "An unfinished"
    assert "the reply was cut" in caplog.text


def test_a_reply_within_limits_raises_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    fine = reply("Hello.", prompt_eval_count=100, eval_count=5, done_reason="stop")

    model(lambda request: fine).generate("Hi")

    assert caplog.text == ""
