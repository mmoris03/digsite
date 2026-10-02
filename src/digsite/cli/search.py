"""The search command."""

import argparse
import sys
from contextlib import closing

from digsite.cli.common import (
    Subparsers,
    authority_options,
    bm25_params,
    corpus_options,
    corpus_path,
    expansion_options,
    expansion_settings,
    fail,
    fusion_options,
    missing_corpus,
    ranking_options,
)
from digsite.search.corpus import CorpusSearch, MissingIndexError, SearchMode
from digsite.search.results import DocumentResult
from digsite.store import connect

_SNIPPET_LENGTH = 160


def register(commands: Subparsers) -> None:
    search = commands.add_parser(
        "search",
        parents=[
            corpus_options(),
            ranking_options(),
            expansion_options(),
            fusion_options(),
            authority_options(),
        ],
        help="search the indexed documents",
    )
    search.add_argument("query", nargs="+", help="what to look for")
    search.add_argument(
        "--mode",
        choices=[mode.value for mode in SearchMode],
        help="match words (lexical), meaning (semantic) or both (hybrid) "
        "(default: hybrid if the corpus has embeddings, lexical otherwise)",
    )
    search.add_argument(
        "--limit", type=int, default=10, help="maximum number of results (default: 10)"
    )
    search.set_defaults(handler=_search)


def _search(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1

    query = " ".join(args.query)
    with closing(connect(path)) as connection:
        corpus = CorpusSearch(
            connection,
            bm25=bm25_params(args),
            expansion=expansion_settings(args),
            rank_constant=args.rank_constant,
            authority_weight=args.authority_weight,
        )
        if corpus.chunks.count() == 0:
            return fail("the corpus has no index; run 'digsite index' first")
        if corpus.chunks.document_count() != corpus.documents.stats().unique:
            print(
                "warning: the documents changed since the index was built; "
                "run 'digsite index' to refresh it",
                file=sys.stderr,
            )

        mode = SearchMode(args.mode) if args.mode else corpus.default_mode
        if args.expand and mode is SearchMode.SEMANTIC:
            return fail("--expand does not apply to semantic search")
        if args.authority_weight and mode is not SearchMode.HYBRID:
            return fail("--authority-weight only applies to hybrid search")
        if not args.mode and mode is SearchMode.LEXICAL:
            print("note: the corpus has no embeddings; matching words only", file=sys.stderr)

        try:
            results = corpus.find_documents(query, mode, args.limit)
        except MissingIndexError as error:
            return fail(str(error))
        if args.expand:
            typed = set(corpus.lexical.analyzer.analyze(query))
            added = [term for term in corpus.lexical.query_terms(query) if term not in typed]
            print(f"expanded with: {', '.join(added) if added else '(nothing)'}")
    if not results:
        print("no results")
        return 0
    indent = " " * 13
    for rank, result in enumerate(results, start=1):
        print(f"{rank:>2}. {result.score:>7.3f}  {result.title}")
        print(f"{indent}{result.url}")
        print(f"{indent}{_snippet(result)}")
    return 0


def _snippet(result: DocumentResult) -> str:
    """One line that shows where in the document the passage is and how it starts."""
    text = " ".join(result.passage.split())
    if len(text) > _SNIPPET_LENGTH:
        text = text[:_SNIPPET_LENGTH].rsplit(" ", 1)[0] + "…"
    return f"[{result.section}] {text}" if result.section else text
