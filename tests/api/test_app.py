import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cli.corpora import CACHING, LOGGING, SITE, make_corpus
from digsite.answer import AnswerSettings
from digsite.api import create_app
from digsite.cli import main
from digsite.embedding import HashingEmbedder
from digsite.library import BuildQueue, Library
from digsite.llm import LanguageModelError
from fakes import ScriptedModel
from library.sites import GUIDE, guide_site

ANSWERED = {"answerable": True, "answer": "The entry used least recently is removed [1]."}
DECLINED = {"answerable": False, "answer": ""}


def build_corpus(data_dir: Path, collection: str = "corpus", *, embeddings: bool = True) -> None:
    """A collection of two documents, indexed in English, with the model-free embedder if any."""
    make_corpus(data_dir, {"/caching": CACHING, "/logging": LOGGING}, collection=collection)
    where = ["--data-dir", str(data_dir), "--collection", collection]
    main(["ingest", *where])
    embedding = ["--model", "hashing"] if embeddings else ["--no-embeddings"]
    main(["index", *where, "--language", "english", *embedding])


class Server:
    """The application over a directory of collections, with a scripted language model."""

    def __init__(self, data_dir: Path, model: ScriptedModel) -> None:
        self.model = model
        self.library = Library(data_dir, embedder_factory=lambda _: HashingEmbedder())
        self.site = guide_site()
        self.builds = BuildQueue(
            self.library,
            embedding_model="hashing",
            client_factory=self.site.new_client,
            delay_seconds=0,
        )
        app = create_app(self.library, self.builds, model, settings=AnswerSettings(rewrites=0))
        self.client = TestClient(app)

    def close(self) -> None:
        self.client.close()
        self.library.close()


type Starter = Callable[..., Server]


@pytest.fixture
def data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    build_corpus(tmp_path)
    capsys.readouterr()
    return tmp_path


@pytest.fixture
def serve(data_dir: Path) -> Iterator[Starter]:
    """Returns a function that starts the application with a model scripted to reply as given."""
    started: list[Server] = []

    def start(*replies: object, directory: Path = data_dir) -> Server:
        server = Server(directory, ScriptedModel(*replies))
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()


def search(server: Server, **params: object) -> dict[str, object]:
    response = server.client.get("/api/search", params={"collection": "corpus", **params})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


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


def test_the_status_says_what_the_server_holds(serve: Starter) -> None:
    status = serve().client.get("/api/status").json()

    assert status == {"collections": 1, "building": False, "language_model": "scripted"}


def test_the_collections_are_described(serve: Starter) -> None:
    body = serve().client.get("/api/collections").json()

    assert body["builds"] == []
    assert body["collections"] == [
        {
            "id": "corpus",
            # Built from stored pages, not crawled: it has no description.
            "title": "corpus",
            "source": "",
            "documents": 2,
            "chunks": 2,
            "modes": ["lexical", "semantic", "hybrid"],
            "default_mode": "hybrid",
            "index_up_to_date": True,
        }
    ]


def test_search_returns_documents_with_their_best_passage(serve: Starter) -> None:
    body = search(serve(), q="  least   recently used ")

    assert body["collection"] == "corpus"
    assert body["query"] == "least recently used"
    assert body["mode"] == "hybrid"
    results = body["results"]
    assert isinstance(results, list)
    first = results[0]
    assert first["url"] == f"{SITE}/caching"
    assert "least recently" in first["passage"]
    assert set(first) == {"url", "title", "section", "passage", "score"}


def test_search_takes_a_mode_and_a_limit(serve: Starter) -> None:
    server = serve()

    lexical = search(server, q="zeppelin", mode="lexical")
    semantic = search(server, q="cache", mode="semantic", limit=1)

    assert lexical["mode"] == "lexical"
    assert lexical["results"] == []
    assert len(semantic["results"]) == 1  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "params",
    [
        {"collection": "corpus"},
        {"collection": "corpus", "q": ""},
        {"collection": "corpus", "q": "   "},
        {"collection": "corpus", "q": "x" * 501},
        {"collection": "corpus", "q": "cache", "mode": "telepathic"},
        {"collection": "corpus", "q": "cache", "limit": 0},
        {"collection": "corpus", "q": "cache", "limit": 51},
        {"q": "cache"},
    ],
)
def test_search_rejects_requests_it_cannot_serve(serve: Starter, params: dict[str, object]) -> None:
    assert serve().client.get("/api/search", params=params).status_code == 422


@pytest.mark.parametrize("collection", ["elsewhere", "../corpus", "Corpus"])
def test_an_unknown_collection_is_not_found(serve: Starter, collection: str) -> None:
    server = serve()

    searched = server.client.get("/api/search", params={"collection": collection, "q": "cache"})
    asked = server.client.post("/api/ask", json={"collection": collection, "question": "cache"})

    assert searched.status_code == asked.status_code == 404
    assert server.model.calls == []


def test_ask_answers_with_the_passages_it_cites(serve: Starter) -> None:
    server = serve(ANSWERED)

    response = server.client.post(
        "/api/ask", json={"collection": "corpus", "question": "what is removed from a full cache"}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["collection"] == "corpus"
    assert body["answered"] is True
    assert body["answer"] == "The entry used least recently is removed [1]."
    assert body["cited"] == [1]
    assert body["passages"][0]["url"].startswith(SITE)
    assert body["queries"] == ["what is removed from a full cache"]
    assert "Question: what is removed from a full cache" in server.model.calls[0].prompt


def test_ask_says_when_the_documents_do_not_answer(serve: Starter) -> None:
    body = (
        serve(DECLINED)
        .client.post("/api/ask", json={"collection": "corpus", "question": "who won the league"})
        .json()
    )

    assert body["answered"] is False
    assert body["answer"] == ""
    assert body["cited"] == []
    assert body["passages"]


def test_ask_reports_a_language_model_that_cannot_be_reached(serve: Starter) -> None:
    server = serve(LanguageModelError("cannot reach Ollama at http://localhost:11434/api/chat"))

    response = server.client.post("/api/ask", json={"collection": "corpus", "question": "cache"})

    assert response.status_code == 503
    assert "cannot reach Ollama" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"collection": "corpus"},
        {"collection": "corpus", "question": ""},
        {"collection": "corpus", "question": "  "},
        {"collection": "corpus", "question": "x" * 501},
        {"collection": "corpus", "question": "a", "x": 1},
        {"question": "cache"},
    ],
)
def test_ask_rejects_requests_it_cannot_serve(serve: Starter, body: dict[str, object]) -> None:
    server = serve()

    assert server.client.post("/api/ask", json=body).status_code == 422
    assert server.model.calls == []


def test_a_website_can_be_added_and_searched_once_built(serve: Starter) -> None:
    server = serve()

    response = server.client.post(
        "/api/collections", json={"url": GUIDE, "max_pages": 10, "language": "english"}
    )

    assert response.status_code == 202
    build = response.json()
    assert build["collection"] == "docs-example-com-guide"
    assert build["stage"] == "queued"
    assert server.builds.wait(build["id"], timeout=30).error is None
    listed = server.client.get("/api/collections").json()
    assert [collection["id"] for collection in listed["collections"]] == [
        "corpus",
        "docs-example-com-guide",
    ]
    assert listed["builds"][0]["stage"] == "done"
    found = server.client.get(
        "/api/search", params={"collection": "docs-example-com-guide", "q": "evicted"}
    ).json()
    assert found["results"][0]["url"] == f"{GUIDE}caching.html"


def test_a_failed_build_is_listed_with_its_reason(serve: Starter) -> None:
    server = serve()

    build = server.client.post("/api/collections", json={"url": f"{SITE}/nowhere/"}).json()
    server.builds.wait(build["id"], timeout=30)

    (listed,) = server.client.get("/api/collections").json()["builds"]
    assert listed["stage"] == "failed"
    assert "no page could be downloaded" in listed["error"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"url": ""},
        {"url": "docs.example.com"},
        {"url": "javascript:alert(1)"},
        {"url": GUIDE, "max_pages": 0},
        {"url": GUIDE, "max_pages": 501},
        {"url": GUIDE, "language": "klingon"},
        {"url": GUIDE, "depth": 3},
    ],
)
def test_a_website_must_be_asked_for_properly(serve: Starter, body: dict[str, object]) -> None:
    server = serve()

    assert server.client.post("/api/collections", json=body).status_code == 422
    assert server.builds.statuses() == []


def test_a_form_on_another_website_cannot_add_one(serve: Starter) -> None:
    # A page elsewhere can post a form to this server, but not with a JSON content
    # type: that needs a preflight request, which the server does not allow.
    server = serve()

    response = server.client.post(
        "/api/collections",
        content=json.dumps({"url": GUIDE}),
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 422
    assert server.builds.statuses() == []


def test_an_empty_directory_serves_no_collections(serve: Starter, tmp_path: Path) -> None:
    server = serve(directory=tmp_path / "empty")

    assert server.client.get("/api/collections").json() == {"collections": [], "builds": []}
    assert server.client.get("/api/status").json()["collections"] == 0


def test_a_collection_without_embeddings_offers_only_lexical_search(
    serve: Starter, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_corpus(data_dir, "words", embeddings=False)
    capsys.readouterr()
    server = serve()

    (described,) = [
        collection
        for collection in server.client.get("/api/collections").json()["collections"]
        if collection["id"] == "words"
    ]
    semantic = server.client.get(
        "/api/search", params={"collection": "words", "q": "cache", "mode": "semantic"}
    )
    lexical = server.client.get("/api/search", params={"collection": "words", "q": "cache"})

    assert described["modes"] == ["lexical"]
    assert described["default_mode"] == "lexical"
    assert semantic.status_code == 400
    assert "has no embeddings" in semantic.json()["detail"]
    assert lexical.json()["results"][0]["url"] == f"{SITE}/caching"


def test_search_works_while_questions_are_answered(serve: Starter) -> None:
    # Requests run in threads that share one read-only connection per collection.
    server = serve(*[ANSWERED] * 4)

    with ThreadPoolExecutor(max_workers=8) as pool:
        asks = [
            pool.submit(
                server.client.post, "/api/ask", json={"collection": "corpus", "question": "cache"}
            )
            for _ in range(4)
        ]
        searches = [
            pool.submit(
                server.client.get, "/api/search", params={"collection": "corpus", "q": "logs"}
            )
            for _ in range(16)
        ]
        statuses = [future.result().status_code for future in [*asks, *searches]]

    assert statuses == [200] * 20


def test_the_interface_is_described_for_clients(serve: Starter) -> None:
    schema = serve().client.get("/api/openapi.json").json()

    assert set(schema["paths"]) == {
        "/api/status",
        "/api/collections",
        "/api/search",
        "/api/ask",
    }


def test_searching_and_answering_never_write_to_a_collection(
    serve: Starter, data_dir: Path
) -> None:
    path = data_dir / "corpus.db"
    before = path.read_bytes()
    server = serve(ANSWERED)

    search(server, q="cache")
    server.client.post("/api/ask", json={"collection": "corpus", "question": "cache"})
    server.client.get("/api/collections")

    assert path.read_bytes() == before
