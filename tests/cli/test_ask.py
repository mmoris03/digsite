from pathlib import Path

import pytest

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite.answer import Answer, QueryKind, QueryPlan, Source
from digsite.cli import common as common_module
from digsite.cli import main
from digsite.cli.ask import format_answer
from digsite.llm import LanguageModelError
from fakes import ScriptedModel

READING = {
    "language": "English",
    "intent": "What is evicted.",
    "kind": "fact",
    "keywords": "cache eviction policy",
    "english": "cache eviction policy",
}
ANSWERED = {
    "answerable": True,
    "answer": "The entry used least recently is removed [1].",
}
DECLINED = {"answerable": False, "answer": ""}


@pytest.fixture
def data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """A corpus with two documents, indexed in English with the model-free embedder."""
    make_corpus(tmp_path, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(
        ["index", "--data-dir", str(tmp_path), "--language", "english", "--model", "hashing",
         "--max-chars", "200"]
    )  # fmt: skip
    capsys.readouterr()
    return str(tmp_path)


class ModelFactory:
    """Stands in for the Ollama client: hands out a scripted model and keeps how it was built."""

    def __init__(self, *replies: object) -> None:
        self.model = ScriptedModel(*replies)
        self.built: list[tuple[str, str]] = []

    def __call__(self, name: str, *, url: str) -> ScriptedModel:
        self.built.append((name, url))
        return self.model


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Returns a function that scripts the replies of the model `ask` will use."""

    def script(*replies: object) -> ModelFactory:
        factory = ModelFactory(*replies)
        monkeypatch.setattr(common_module, "create_language_model", factory)
        return factory

    return script


def test_ask_prints_the_answer_and_the_sources_it_cites(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted(READING, ANSWERED)

    exit_code = main(["ask", "what is removed when the cache is full", "--data-dir", data_dir])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert exit_code == 0
    assert lines[0] == "The entry used least recently is removed [1]."
    assert lines[1] == ""
    assert lines[2] == "Sources:"
    assert lines[3] == " [1] Caching"
    assert lines[4].strip() == f"{SITE}/caching"
    assert len(lines) == 5
    assert "warning" not in captured.err


def test_ask_uses_the_model_and_server_it_is_told(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = scripted(READING, ANSWERED, ANSWERED)

    main(["ask", "cache", "--data-dir", data_dir])
    main(
        ["ask", "cache", "--llm", "qwen2.5:3b", "--llm-url", "http://gpu-box:11434",
         "--rewrites", "0", "--data-dir", data_dir]
    )  # fmt: skip

    assert factory.built == [
        ("gemma3:4b", "http://localhost:11434"),
        ("qwen2.5:3b", "http://gpu-box:11434"),
    ]


def test_the_environment_sets_the_default_model_and_server(
    data_dir: str, scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = scripted(ANSWERED, ANSWERED)
    monkeypatch.setenv("DIGSITE_LLM", "qwen2.5:3b")
    monkeypatch.setenv("DIGSITE_LLM_URL", "http://ollama:11434")

    main(["ask", "cache", "--rewrites", "0", "--data-dir", data_dir])
    main(["ask", "cache", "--rewrites", "0", "--llm", "llama3.2:3b", "--data-dir", data_dir])

    # An option on the command line still wins over the environment.
    assert factory.built == [
        ("qwen2.5:3b", "http://ollama:11434"),
        ("llama3.2:3b", "http://ollama:11434"),
    ]


def test_ask_without_rewrites_calls_the_model_once(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = scripted(ANSWERED)

    exit_code = main(["ask", "full cache", "--rewrites", "0", "--data-dir", data_dir])

    assert exit_code == 0
    assert len(factory.model.calls) == 1


def test_ask_says_when_the_documents_do_not_answer(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted(READING, DECLINED)

    exit_code = main(["ask", "who won the league", "--data-dir", data_dir])

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert lines[0] == "The documents do not answer this question."
    assert lines[2] == "Closest passages:"
    assert lines[3].startswith(" [1] ")


def test_ask_warns_about_an_answer_without_citations(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted(READING, {"answerable": True, "answer": "The oldest entry goes."})

    exit_code = main(["ask", "full cache", "--data-dir", data_dir])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "The oldest entry goes."
    assert "warning: the answer cites none of its passages" in captured.err


def test_ask_can_show_every_passage_the_model_was_given(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = scripted(READING, ANSWERED)

    main(["ask", "full cache", "--passages", "2", "--show-passages", "--data-dir", data_dir])

    output = capsys.readouterr().out
    assert "Passages given to the model:" in output
    listed = output.split("Passages given to the model:")[1]
    assert " [1] " in listed
    assert " [2] " in listed
    assert " [3] " not in listed
    # The model is given the two passages asked for and no more.
    passages = factory.model.calls[1].prompt.split("Question:")[0]
    assert "\n[2] " in passages
    assert "\n[3] " not in passages


def test_ask_reports_a_model_that_cannot_be_reached(
    data_dir: str, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted(LanguageModelError("cannot reach Ollama at http://localhost:11434/api/chat"))

    exit_code = main(["ask", "full cache", "--data-dir", data_dir])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: cannot reach Ollama" in captured.err
    assert captured.out == ""


def test_ask_needs_an_indexed_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "cache", "--data-dir", str(tmp_path)]) == 1
    assert "run 'digsite crawl' first" in capsys.readouterr().err

    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    assert main(["ask", "cache", "--data-dir", str(tmp_path)]) == 1
    assert "run 'digsite index' first" in capsys.readouterr().err


def test_ask_needs_embeddings_for_the_modes_that_use_them(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path, {"/caching": CACHING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--language", "english", "--no-embeddings"])
    capsys.readouterr()
    scripted(ANSWERED)

    assert main(["ask", "cache", "--mode", "semantic", "--data-dir", str(tmp_path)]) == 1
    assert "has no embeddings" in capsys.readouterr().err
    # Without embeddings the default is to search by words.
    assert main(["ask", "full cache", "--rewrites", "0", "--data-dir", str(tmp_path)]) == 0


def test_ask_rejects_a_number_of_passages_below_one(
    data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ask", "cache", "--passages", "0", "--data-dir", data_dir]) == 1
    assert "--passages must be at least 1" in capsys.readouterr().err


SOURCES = (
    Source(1, 11, f"{SITE}/caching", "Caching", "Eviction", "The least recently used goes."),
    Source(2, 12, f"{SITE}/caching", "Caching", "", "The cache stores results."),
    Source(3, 27, f"{SITE}/logging", "Logging", "Rotation", "Logs rotate daily.\nKept 30 days."),
)
PLAN = QueryPlan("q", "", "", QueryKind.FACT, ("q",))


def test_the_sources_are_listed_in_the_order_the_answer_cites_them() -> None:
    answer = Answer("q", True, "Daily [3]. The oldest goes [1].", SOURCES, (3, 1), PLAN)

    assert format_answer(answer) == (
        "Daily [3]. The oldest goes [1].\n"
        "\n"
        "Sources:\n"
        f" [3] Logging > Rotation\n     {SITE}/logging\n"
        f" [1] Caching > Eviction\n     {SITE}/caching"
    )


def test_a_declined_answer_lists_the_closest_passages() -> None:
    answer = Answer("q", False, "", SOURCES, (), PLAN)

    assert format_answer(answer) == (
        "The documents do not answer this question.\n"
        "\n"
        "Closest passages:\n"
        f" [1] Caching > Eviction\n     {SITE}/caching\n"
        f" [2] Caching\n     {SITE}/caching\n"
        f" [3] Logging > Rotation\n     {SITE}/logging"
    )


def test_a_declined_answer_with_nothing_found_says_only_that() -> None:
    answer = Answer("q", False, "", (), (), PLAN)

    assert format_answer(answer) == "The documents do not answer this question."


def test_passages_are_shown_with_their_text_indented() -> None:
    answer = Answer("q", True, "Daily [3].", SOURCES[2:], (3,), PLAN)

    assert format_answer(answer, show_passages=True).endswith(
        "Passages given to the model:\n"
        f" [3] Logging > Rotation\n     {SITE}/logging\n"
        "      Logs rotate daily.\n"
        "      Kept 30 days."
    )
