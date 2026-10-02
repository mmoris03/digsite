import json
from pathlib import Path

import pytest

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite.cli import common as common_module
from digsite.cli import main
from digsite.llm import LanguageModelError
from fakes import ScriptedModel

QUERIES = f"""what is removed when the cache is full\t{SITE}/caching
how long are logs kept\t{SITE}/logging
"""
UNANSWERABLE = "# not in the corpus\nwho won the league\n"

CITES_FIRST = {"answerable": True, "answer": "As the first passage says [1]."}
CITES_NOTHING = {"answerable": True, "answer": "Thirty days."}
DECLINED = {"answerable": False, "answer": ""}


@pytest.fixture
def data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A corpus with two documents, indexed in English with the model-free embedder."""
    make_corpus(tmp_path, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(tmp_path)])
    main(["index", "--data-dir", str(tmp_path), "--language", "english", "--model", "hashing"])
    (tmp_path / "queries.tsv").write_text(QUERIES, encoding="utf-8")
    (tmp_path / "unanswerable.tsv").write_text(UNANSWERABLE, encoding="utf-8")
    capsys.readouterr()
    return tmp_path


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Returns a function that scripts the replies of the model the command will use."""

    def script(*replies: object) -> ScriptedModel:
        model = ScriptedModel(*replies)
        monkeypatch.setattr(common_module, "create_language_model", lambda name, *, url: model)
        return model

    return script


def arguments(data_dir: Path, *extra: str) -> list[str]:
    queries = str(data_dir / "queries.tsv")
    return ["eval", "answers", "--queries", queries, "--rewrites", "0", *extra,
            "--data-dir", str(data_dir)]  # fmt: skip


def read_report(output: str) -> dict[str, float]:
    rows = {}
    for line in output.splitlines():
        if line.startswith("  "):
            name, value = line.rsplit(maxsplit=1)
            rows[name.strip()] = float(value)
    return rows


def test_answers_are_scored_by_the_pages_they_cite(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    # Lexical search puts the right page first for both questions.
    scripted(CITES_FIRST, CITES_NOTHING)

    exit_code = main(arguments(data_dir, "--mode", "lexical"))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.splitlines()[0] == "2 questions that the corpus answers"
    assert read_report(output) == {
        "answered": 1.0,
        "cites a relevant page": 0.5,
        "cites nothing": 0.5,
        "citation precision": 1.0,
        "relevant page found": 1.0,
    }


def test_questions_without_an_answer_are_scored_by_how_many_are_declined(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted(CITES_FIRST, CITES_FIRST, DECLINED)
    unanswerable = str(data_dir / "unanswerable.tsv")

    exit_code = main(arguments(data_dir, "--unanswerable", unanswerable))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1 questions that the corpus does not answer" in output
    assert read_report(output)["declined"] == 1.0


def test_the_limit_shortens_both_sets(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    model = scripted(CITES_FIRST, DECLINED)
    unanswerable = str(data_dir / "unanswerable.tsv")

    main(arguments(data_dir, "--unanswerable", unanswerable, "--limit", "1"))

    assert capsys.readouterr().out.startswith("1 questions that the corpus answers")
    assert len(model.calls) == 2


def test_answers_are_written_to_a_file_as_they_come(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted(CITES_FIRST, DECLINED)
    output = data_dir / "runs" / "answers.jsonl"

    main(arguments(data_dir, "--output", str(output)))

    header, first, second = (json.loads(line) for line in output.read_text("utf-8").splitlines())
    assert header == {"settings": {"llm": "gemma3:4b", "mode": None, "passages": 6, "rewrites": 0}}
    assert first["question"] == "what is removed when the cache is full"
    assert first["answered"] is True
    assert first["cited"] == [1]
    assert first["sources"][0]["url"].startswith(SITE)
    assert second["answered"] is False


def test_a_run_that_stopped_is_resumed_without_asking_again(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    output = data_dir / "answers.jsonl"
    scripted(CITES_FIRST, LanguageModelError("cannot reach Ollama"))

    assert main(arguments(data_dir, "--output", str(output))) == 1
    assert "cannot reach Ollama" in capsys.readouterr().err
    assert len(output.read_text("utf-8").splitlines()) == 2

    model = scripted(CITES_FIRST)
    assert main(arguments(data_dir, "--output", str(output))) == 0

    # Only the question that had no answer yet was asked.
    assert len(model.calls) == 1
    assert "how long are logs kept" in model.calls[0].prompt
    assert len(output.read_text("utf-8").splitlines()) == 3
    assert read_report(capsys.readouterr().out)["answered"] == 1.0


def test_answers_obtained_with_other_settings_are_not_mixed_in(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    output = data_dir / "answers.jsonl"
    scripted(CITES_FIRST, CITES_FIRST)
    main(arguments(data_dir, "--output", str(output)))
    capsys.readouterr()

    exit_code = main(arguments(data_dir, "--output", str(output), "--passages", "3"))

    assert exit_code == 1
    assert "holds answers obtained with other settings" in capsys.readouterr().err


def test_an_output_file_that_is_something_else_is_not_overwritten(
    data_dir: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    output = data_dir / "notes.txt"
    output.write_text("my notes\n", encoding="utf-8")
    scripted()

    exit_code = main(arguments(data_dir, "--output", str(output)))

    assert exit_code == 1
    assert "is not the output of an earlier run" in capsys.readouterr().err
    assert output.read_text("utf-8") == "my notes\n"


def test_what_is_missing_is_reported(
    data_dir: Path, tmp_path_factory: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path_factory.mktemp("empty")
    assert main(["eval", "answers", "--queries", "q.tsv", "--data-dir", str(empty)]) == 1
    assert "run 'digsite crawl' first" in capsys.readouterr().err

    assert main(["eval", "answers", "--queries", "missing.tsv", "--data-dir", str(data_dir)]) == 1
    assert "missing.tsv does not exist" in capsys.readouterr().err

    assert main(arguments(data_dir, "--unanswerable", "missing.tsv")) == 1
    assert "missing.tsv does not exist" in capsys.readouterr().err

    assert main(arguments(data_dir, "--passages", "0")) == 1
    assert "--passages must be at least 1" in capsys.readouterr().err

    (data_dir / "other.tsv").write_text(f"cache\t{SITE}/not-crawled\n", encoding="utf-8")
    other = ["eval", "answers", "--queries", str(data_dir / "other.tsv")]
    assert main([*other, "--data-dir", str(data_dir)]) == 1
    assert "not in the corpus" in capsys.readouterr().err
