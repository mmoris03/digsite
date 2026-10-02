from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from cli.corpora import CACHING, LOGGING, make_corpus
from digsite.cli import main


class FakeServer:
    """Stands in for uvicorn.run: checks the application while the corpus is open."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.status: dict[str, object] = {}

    def __call__(self, app: object, *, host: str, port: int, log_level: str) -> None:
        self.calls.append((host, port))
        with TestClient(app) as client:  # type: ignore[arg-type]
            self.status = client.get("/api/status").json()


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    server = FakeServer()
    monkeypatch.setattr(uvicorn, "run", server)
    return server


def index(data_dir: Path, *options: str) -> None:
    make_corpus(data_dir, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(data_dir)])
    main(["index", "--data-dir", str(data_dir), "--language", "english", *options])


def test_serve_starts_the_application_on_this_computer_by_default(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--model", "hashing")
    capsys.readouterr()

    exit_code = main(["serve", "--data-dir", str(tmp_path)])

    assert exit_code == 0
    assert fake_server.calls == [("127.0.0.1", 8000)]
    assert "at http://127.0.0.1:8000/" in capsys.readouterr().out
    assert fake_server.status["documents"] == 2
    assert fake_server.status["default_mode"] == "hybrid"
    assert fake_server.status["language_model"] == "gemma3:4b"


def test_serve_takes_the_address_the_model_and_the_mode(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--model", "hashing")

    main(["serve", "--host", "0.0.0.0", "--port", "9000", "--llm", "qwen2.5:3b",
          "--mode", "semantic", "--data-dir", str(tmp_path)])  # fmt: skip

    assert fake_server.calls == [("0.0.0.0", 9000)]
    assert fake_server.status["language_model"] == "qwen2.5:3b"
    assert fake_server.status["default_mode"] == "semantic"


def test_serve_without_embeddings_searches_by_words(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--no-embeddings")

    assert main(["serve", "--data-dir", str(tmp_path)]) == 0
    assert fake_server.status["default_mode"] == "lexical"


def test_serve_needs_an_indexed_corpus(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["serve", "--data-dir", str(tmp_path)]) == 1
    assert "run 'digsite crawl' first" in capsys.readouterr().err

    make_corpus(tmp_path, {"/caching": CACHING})
    assert main(["serve", "--data-dir", str(tmp_path)]) == 1
    assert "run 'digsite index' first" in capsys.readouterr().err
    assert fake_server.calls == []


def test_serve_needs_embeddings_for_a_mode_that_uses_them(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--no-embeddings")
    capsys.readouterr()

    assert main(["serve", "--mode", "hybrid", "--data-dir", str(tmp_path)]) == 1
    assert "has no embeddings" in capsys.readouterr().err


def test_serve_rejects_a_number_of_passages_below_one(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--no-embeddings")
    capsys.readouterr()

    assert main(["serve", "--passages", "0", "--data-dir", str(tmp_path)]) == 1
    assert "--passages must be at least 1" in capsys.readouterr().err
