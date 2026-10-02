from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from cli.corpora import CACHING, LOGGING, make_corpus
from digsite.cli import main


class FakeServer:
    """Stands in for uvicorn.run: looks at the application while the server would run."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.status: dict[str, object] = {}
        self.collections: list[dict[str, object]] = []

    def __call__(self, app: object, *, host: str, port: int, log_level: str) -> None:
        self.calls.append((host, port))
        with TestClient(app) as client:  # type: ignore[arg-type]
            self.status = client.get("/api/status").json()
            self.collections = client.get("/api/collections").json()["collections"]


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    server = FakeServer()
    monkeypatch.setattr(uvicorn, "run", server)
    return server


def index(data_dir: Path, *options: str, collection: str = "corpus") -> None:
    make_corpus(data_dir, {"/caching": CACHING, "/logging": LOGGING}, collection=collection)
    where = ["--data-dir", str(data_dir), "--collection", collection]
    main(["ingest", *where])
    main(["index", *where, "--language", "english", *options])


def test_serve_starts_on_this_computer_with_the_collections_of_the_directory(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--model", "hashing")
    index(tmp_path, "--no-embeddings", collection="words")
    capsys.readouterr()

    exit_code = main(["serve", "--data-dir", str(tmp_path)])

    assert exit_code == 0
    assert fake_server.calls == [("127.0.0.1", 8000)]
    assert "at http://127.0.0.1:8000/" in capsys.readouterr().out
    assert fake_server.status["language_model"] == "gemma3:4b"
    modes = {collection["id"]: collection["default_mode"] for collection in fake_server.collections}
    assert modes == {"corpus": "hybrid", "words": "lexical"}


def test_serve_starts_with_no_collections_so_that_websites_can_be_added(
    tmp_path: Path, fake_server: FakeServer
) -> None:
    data_dir = tmp_path / "new"

    assert main(["serve", "--data-dir", str(data_dir)]) == 0
    assert data_dir.is_dir()
    assert fake_server.status["collections"] == 0


def test_serve_takes_the_address_the_model_and_the_mode(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    index(tmp_path, "--model", "hashing")

    main(["serve", "--host", "0.0.0.0", "--port", "9000", "--llm", "qwen2.5:3b",
          "--mode", "semantic", "--data-dir", str(tmp_path)])  # fmt: skip

    assert fake_server.calls == [("0.0.0.0", 9000)]
    assert fake_server.status["language_model"] == "qwen2.5:3b"
    assert fake_server.collections[0]["default_mode"] == "semantic"


def test_serve_removes_what_an_interrupted_build_left(
    tmp_path: Path, fake_server: FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "half-done.db.partial").write_bytes(b"")

    assert main(["serve", "--data-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "half-done.db.partial").exists()
    assert "removed half-done" in caplog.text


def test_serve_rejects_a_number_of_passages_below_one(
    tmp_path: Path, fake_server: FakeServer, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["serve", "--passages", "0", "--data-dir", str(tmp_path)]) == 1
    assert "--passages must be at least 1" in capsys.readouterr().err
    assert fake_server.calls == []
