"""The HTTP interface: search and answers as JSON, and a page to use them from a browser."""

import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from digsite import __version__
from digsite.answer import Answerer, AnswerSettings
from digsite.api.schemas import (
    MAX_QUERY_LENGTH,
    AskRequest,
    AskResponse,
    SearchResponse,
    SearchResult,
    Status,
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
    corpus: CorpusSearch,
    model: LanguageModel,
    *,
    default_mode: SearchMode | None = None,
    settings: AnswerSettings | None = None,
) -> FastAPI:
    """Build the web application over an indexed corpus.

    The application only reads the corpus. It is handed everything it uses,
    already built: the server and the tests decide what that is.

    Args:
        corpus: The corpus, on a connection that the request threads can share.
        model: The language model that answers questions. Search works
            without it.
        default_mode: How to search when a request does not say.
        settings: How questions are answered.
    """
    mode_by_default = default_mode or corpus.default_mode
    page = (_STATIC / "index.html").read_text(encoding="utf-8")

    app = FastAPI(
        title="Digsite",
        version=__version__,
        summary="Hybrid search and answers with sources over a crawled corpus",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        return HTMLResponse(page, headers=_PAGE_HEADERS)

    @app.get("/api/status")
    def status() -> Status:
        chunks = corpus.chunks.count()
        documents = corpus.documents.stats().unique
        modes = list(SearchMode) if corpus.has_embeddings else [SearchMode.LEXICAL]
        return Status(
            documents=documents,
            chunks=chunks,
            index_up_to_date=corpus.chunks.document_count() == documents,
            embedding_model=corpus.chunks.embedding_model(),
            modes=modes,
            default_mode=mode_by_default,
            language_model=model.name,
        )

    @app.get("/api/search")
    def search(
        q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_LENGTH)],
        mode: SearchMode | None = None,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> SearchResponse:
        """Find the documents that best match a query, each shown by its best passage."""
        query = " ".join(q.split())
        if not query:
            raise HTTPException(422, "the query is empty")
        chosen = mode or mode_by_default
        try:
            results = corpus.find_documents(query, chosen, limit)
        except MissingIndexError as error:
            raise HTTPException(400, str(error)) from error
        return SearchResponse(
            query=query, mode=chosen, results=[SearchResult.of(result) for result in results]
        )

    @app.post("/api/ask")
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question from the documents, citing the passages it comes from.

        On a computer without a GPU this takes one to two minutes.
        """
        try:
            answerer = Answerer.for_corpus(
                corpus, model, mode=request.mode or mode_by_default, settings=settings
            )
        except MissingIndexError as error:
            raise HTTPException(400, str(error)) from error
        started = time.perf_counter()
        try:
            answer = answerer.ask(request.question)
        except LanguageModelError as error:
            raise HTTPException(503, str(error)) from error
        return AskResponse.of(answer, seconds=time.perf_counter() - started)

    return app
