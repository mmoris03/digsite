"""Command-line interface."""

import argparse
import logging

from digsite import __version__
from digsite.cli import ask, corpus, evaluate, search, serve

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "trafilatura",
    "htmldate",
    "courlan",
    "sentence_transformers",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digsite", description="Hybrid search engine and RAG pipeline"
    )
    parser.add_argument("--version", action="version", version=f"digsite {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    corpus.register(commands)
    search.register(commands)
    ask.register(commands)
    serve.register(commands)
    evaluate.register(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    exit_code: int = args.handler(args)
    return exit_code
