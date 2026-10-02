"""The HTTP interface: collections, search and answers as JSON, and a page to use them."""

import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from digsite import __version__
from digsite.answer import Answerer, AnswerSettings
from digsite.api.schemas import (
    MAX_QUERY_LENGTH,
    AddCollectionRequest,
    AskRequest,
    AskResponse,
    Build,
    Collection,
    Collections,
    SearchResponse,
    SearchResult,
    Status,
)
from digsite.library import (
    BuildError,
    BuildQueue,
    BuildRequest,
    Library,
    UnknownCollectionError,
    crawl_scope,
)
from digsite.llm import LanguageModel, LanguageModelError
from digsite.search.corpus import CorpusSearch, MissingIndexError, SearchMode

_STATIC = Path(__file__).parent / "static"

# The page loads its script and style from this server and nothing else.
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def create_app(
    library: Library,
    builds: BuildQueue,
    model: LanguageModel,
    *,
    default_mode: SearchMode | None = None,
    settings: AnswerSettings | None = None,
) -> FastAPI:
    """Build the web application over a library of collections.

    It is handed everything it uses, already built: the server and the tests
    decide what that is. Searching and answering only read the collections;
    adding one goes through `builds`, which writes new files in the background.

    Args:
        library: The collections.
        builds: Where new collections are asked for.
        model: The language model that answers questions. Search works
            without it.
        default_mode: How to search when a request does not say, if the
            collection allows it.
        settings: How questions are answered.
    """
    page = (_STATIC / "index.html").read_text(encoding="utf-8")

    app = FastAPI(
        title="Digsite",
        version=__version__,
        summary="Hybrid search and answers with sources over crawled websites",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    def open_collection(collection_id: str) -> CorpusSearch:
        try:
            return library.corpus(collection_id)
        except UnknownCollectionError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        except MissingIndexError as error:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    def mode_for(corpus: CorpusSearch, requested: SearchMode | None) -> SearchMode:
        if requested is not None:
            return requested
        # The server's default, unless the collection was indexed without embeddings.
        if default_mode is SearchMode.LEXICAL or (default_mode and corpus.has_embeddings):
            return default_mode
        return corpus.default_mode

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        return HTMLResponse(page, headers=_PAGE_HEADERS)

    @app.get("/api/status")
    def server_status() -> Status:
        return Status(
            collections=len(library.summaries()),
            building=any(build.active for build in builds.statuses()),
            language_model=model.name,
        )

    @app.get("/api/collections")
    def list_collections() -> Collections:
        """The collections that can be searched, and the websites being added."""
        return Collections(
            collections=[Collection.of(summary, default_mode) for summary in library.summaries()],
            builds=[Build.of(build) for build in builds.statuses()],
        )

    @app.post("/api/collections", status_code=status.HTTP_202_ACCEPTED)
    def add_collection(request: AddCollectionRequest) -> Build:
        """Crawl a website and make it a collection, in the background.

        The build is queued behind any other. Follow it in `GET /api/collections`:
        crawling waits a second between pages, and computing embeddings on a
        CPU takes a few minutes for a hundred pages.
        """
        try:
            crawl_scope(request.url)
        except BuildError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        build = builds.submit(BuildRequest(request.url, request.max_pages, request.language))
        return Build.of(build)

    @app.get("/api/search")
    def search(
        collection: Annotated[str, Query(min_length=1, description="Id of the collection")],
        q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_LENGTH)],
        mode: SearchMode | None = None,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> SearchResponse:
        """Find the documents that best match a query, each shown by its best passage."""
        query = " ".join(q.split())
        if not query:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "the query is empty")
        corpus = open_collection(collection)
        chosen = mode_for(corpus, mode)
        try:
            results = corpus.find_documents(query, chosen, limit)
        except MissingIndexError as error:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
        return SearchResponse(
            collection=collection,
            query=query,
            mode=chosen,
            results=[SearchResult.of(result) for result in results],
        )

    @app.post("/api/ask")
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question from a collection, citing the passages it comes from.

        On a computer without a GPU this takes one to two minutes.
        """
        corpus = open_collection(request.collection)
        try:
            answerer = Answerer.for_corpus(
                corpus, model, mode=mode_for(corpus, request.mode), settings=settings
            )
        except MissingIndexError as error:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
        started = time.perf_counter()
        try:
            answer = answerer.ask(request.question)
        except LanguageModelError as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
        return AskResponse.of(request.collection, answer, seconds=time.perf_counter() - started)

    return app
