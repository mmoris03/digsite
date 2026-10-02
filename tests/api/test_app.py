from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite.answer import AnswerSettings
from digsite.api import create_app
from digsite.cli import main
from digsite.cli.common import CORPUS_FILENAME
from digsite.llm import LanguageModelError
from digsite.search.corpus import CorpusSearch, SearchMode
from digsite.store import connect
from fakes import ScriptedModel

ANSWERED = {"answerable": True, "answer": "The entry used least recently is removed [1]."}
DECLINED = {"answerable": False, "answer": ""}


def build_corpus(data_dir: Path, *, embeddings: bool = True) -> Path:
    """A corpus of two documents, indexed in English, with the model-free embedder if any."""
    make_corpus(data_dir, {"/caching": CACHING, "/logging": LOGGING})
    main(["ingest", "--data-dir", str(data_dir)])
    embedding = ["--model", "hashing"] if embeddings else ["--no-embeddings"]
    main(["index", "--data-dir", str(data_dir), "--language", "english", *embedding])
    return data_dir / CORPUS_FILENAME


class Server:
    """The application over a corpus, with a scripted language model."""

    def __init__(self, path: Path, model: ScriptedModel) -> None:
        self.model = model
        self._connection = connect(path, read_only=True)
        app = create_app(CorpusSearch(self._connection), model, settings=AnswerSettings(rewrites=0))
        self.client = TestClient(app)

    def close(self) -> None:
        self.client.close()
        self._connection.close()


@pytest.fixture
def corpus_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    path = build_corpus(tmp_path)
    capsys.readouterr()
    return path


type Starter = Callable[..., Server]


@pytest.fixture
def serve(corpus_path: Path) -> Iterator[Starter]:
    """Returns a function that starts the application with a model scripted to reply as given."""
    started: list[Server] = []

    def start(*replies: object) -> Server:
        server = Server(corpus_path, ScriptedModel(*replies))
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()


def test_the_page_is_served_with_a_strict_content_policy(serve: Starter) -> None:
    client = serve().client

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Digsite</title>" in response.text
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "default-src 'none'" in policy
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_the_status_describes_the_corpus(serve: Starter) -> None:
    status = serve().client.get("/api/status").json()

    assert status == {
        "documents": 2,
        "chunks": 2,
        "index_up_to_date": True,
        "embedding_model": "hashing-256",
        "modes": ["lexical", "semantic", "hybrid"],
        "default_mode": "hybrid",
        "language_model": "scripted",
    }


def test_search_returns_documents_with_their_best_passage(serve: Starter) -> None:
    response = serve().client.get("/api/search", params={"q": "  least   recently used "})

    body = response.json()
    assert response.status_code == 200
    assert body["query"] == "least recently used"
    assert body["mode"] == "hybrid"
    first = body["results"][0]
    assert first["url"] == f"{SITE}/caching"
    assert first["title"] == "Caching"
    assert "least recently" in first["passage"]
    assert set(first) == {"url", "title", "section", "passage", "score"}


def test_search_takes_a_mode_and_a_limit(serve: Starter) -> None:
    client = serve().client

    lexical = client.get("/api/search", params={"q": "zeppelin", "mode": "lexical"}).json()
    semantic = client.get("/api/search", params={"q": "cache", "mode": "semantic", "limit": 1})

    assert lexical["mode"] == "lexical"
    assert lexical["results"] == []
    assert len(semantic.json()["results"]) == 1


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 501},
        {"q": "cache", "mode": "telepathic"},
        {"q": "cache", "limit": 0},
        {"q": "cache", "limit": 51},
    ],
)
def test_search_rejects_requests_it_cannot_serve(serve: Starter, params: dict[str, object]) -> None:
    assert serve().client.get("/api/search", params=params).status_code == 422


def test_ask_answers_with_the_passages_it_cites(serve: Starter) -> None:
    server = serve(ANSWERED)

    response = server.client.post(
        "/api/ask", json={"question": "what is removed from a full cache"}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["answered"] is True
    assert body["answer"] == "The entry used least recently is removed [1]."
    assert body["cited"] == [1]
    assert body["passages"][0]["number"] == 1
    assert body["passages"][0]["url"].startswith(SITE)
    assert body["queries"] == ["what is removed from a full cache"]
    assert body["seconds"] >= 0
    assert "Question: what is removed from a full cache" in server.model.calls[0].prompt


def test_ask_says_when_the_documents_do_not_answer(serve: Starter) -> None:
    body = serve(DECLINED).client.post("/api/ask", json={"question": "who won the league"}).json()

    assert body["answered"] is False
    assert body["answer"] == ""
    assert body["cited"] == []
    assert body["passages"]


def test_ask_reports_a_language_model_that_cannot_be_reached(serve: Starter) -> None:
    server = serve(LanguageModelError("cannot reach Ollama at http://localhost:11434/api/chat"))

    response = server.client.post("/api/ask", json={"question": "full cache"})

    assert response.status_code == 503
    assert "cannot reach Ollama" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{}, {"question": ""}, {"question": "  "}, {"question": "x" * 501}, {"question": "a", "x": 1}],
)
def test_ask_rejects_requests_it_cannot_serve(serve: Starter, body: dict[str, object]) -> None:
    server = serve()

    assert server.client.post("/api/ask", json=body).status_code == 422
    assert server.model.calls == []


def test_search_works_while_a_question_is_being_answered(corpus_path: Path) -> None:
    # Requests run in threads that share one read-only connection.
    model = ScriptedModel(*[ANSWERED] * 4)
    server = Server(corpus_path, model)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            asks = [
                pool.submit(server.client.post, "/api/ask", json={"question": "cache"})
                for _ in range(4)
            ]
            searches = [
                pool.submit(server.client.get, "/api/search", params={"q": "logs"})
                for _ in range(16)
            ]
            statuses = [future.result().status_code for future in [*asks, *searches]]
    finally:
        server.close()

    assert statuses == [200] * 20


def test_a_corpus_without_embeddings_offers_only_lexical_search(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build_corpus(tmp_path, embeddings=False)
    capsys.readouterr()
    server = Server(path, ScriptedModel())
    try:
        status = server.client.get("/api/status").json()
        semantic = server.client.get("/api/search", params={"q": "cache", "mode": "semantic"})
        lexical = server.client.get("/api/search", params={"q": "cache"})
    finally:
        server.close()

    assert status["modes"] == ["lexical"]
    assert status["default_mode"] == "lexical"
    assert semantic.status_code == 400
    assert "has no embeddings" in semantic.json()["detail"]
    assert lexical.json()["results"][0]["title"] == "Caching"


def test_the_interface_is_described_for_clients(serve: Starter) -> None:
    schema = serve().client.get("/api/openapi.json").json()

    assert set(schema["paths"]) == {"/api/status", "/api/search", "/api/ask"}
    assert SearchMode.HYBRID.value in str(schema["components"])


def test_the_application_never_writes_to_the_corpus(corpus_path: Path) -> None:
    before = corpus_path.read_bytes()
    server = Server(corpus_path, ScriptedModel(ANSWERED))
    try:
        server.client.get("/api/search", params={"q": "cache"})
        server.client.post("/api/ask", json={"question": "cache"})
    finally:
        server.close()

    assert corpus_path.read_bytes() == before
