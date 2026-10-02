"""The serve command."""

import argparse
from contextlib import closing

from digsite.cli.common import (
    Subparsers,
    answer_options,
    answer_settings,
    corpus_options,
    corpus_path,
    fail,
    invalid_answer_options,
    language_model,
    missing_corpus,
    search_mode,
)
from digsite.search.corpus import CorpusSearch, MissingIndexError
from digsite.store import SchemaVersionError, connect


def register(commands: Subparsers) -> None:
    serve = commands.add_parser(
        "serve",
        parents=[corpus_options(), answer_options()],
        help="serve search and answers over HTTP, with a page to use them from a browser",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (default: 127.0.0.1, reachable from this computer only)",
    )
    serve.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    serve.set_defaults(handler=_serve)


def _serve(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    if problem := invalid_answer_options(args):
        return fail(problem)

    # Imported here: only this command needs the web server.
    import uvicorn

    from digsite.api import create_app

    try:
        connection = connect(path, read_only=True)
    except SchemaVersionError as error:
        return fail(str(error))
    with closing(connection):
        corpus = CorpusSearch(connection)
        mode = search_mode(args) or corpus.default_mode
        try:
            # Load the indexes and the embedding model now, not on the first request.
            corpus.retriever(mode)
        except MissingIndexError as error:
            return fail(str(error))
        _ = corpus.page_ids
        app = create_app(
            corpus, language_model(args), default_mode=mode, settings=answer_settings(args)
        )
        print(f"serving {path} at http://{args.host}:{args.port}/ (Ctrl+C to stop)")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
