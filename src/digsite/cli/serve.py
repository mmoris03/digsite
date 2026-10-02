"""The serve command."""

import argparse
import logging
import threading

from digsite.cli.common import (
    Subparsers,
    answer_options,
    answer_settings,
    fail,
    invalid_answer_options,
    language_model,
    library_options,
    search_mode,
)
from digsite.library import BuildQueue, Library

logger = logging.getLogger(__name__)


def register(commands: Subparsers) -> None:
    serve = commands.add_parser(
        "serve",
        parents=[library_options(), answer_options()],
        help="serve the collections over HTTP, with a page to search them, ask questions "
        "and add websites",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on (default: 127.0.0.1, reachable from this computer only)",
    )
    serve.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    serve.set_defaults(handler=_serve)


def _serve(args: argparse.Namespace) -> int:
    if problem := invalid_answer_options(args):
        return fail(problem)

    # Imported here: only this command needs the web server.
    import uvicorn

    from digsite.api import create_app

    library = Library(args.data_dir)
    for collection_id in library.remove_partial_builds():
        logger.warning(
            "removed %s, which was being built when the server last stopped", collection_id
        )
    app = create_app(
        library,
        BuildQueue(library),
        language_model(args),
        default_mode=search_mode(args),
        settings=answer_settings(args),
    )
    # Searches work at once; this only spares the first search of each collection the wait.
    threading.Thread(target=library.load_all, name="load-collections", daemon=True).start()
    print(f"serving {args.data_dir} at http://{args.host}:{args.port}/ (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        library.close()
    return 0
