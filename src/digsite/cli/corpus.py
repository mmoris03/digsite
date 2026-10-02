"""Commands that build the corpus: crawl, ingest, index and stats."""

import argparse
import sys
from contextlib import closing

from digsite.cli.common import (
    Subparsers,
    analysis_options,
    analyzer_settings,
    corpus_options,
    corpus_path,
    embedding_options,
    fail,
    missing_corpus,
)
from digsite.crawl import CrawlConfig, Crawler, CrawlStats, http_client
from digsite.crawl.crawler import DEFAULT_USER_AGENT
from digsite.embedding import create_embedder
from digsite.index.chunking import DEFAULT_MAX_CHARS
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.pipeline import build_authority, build_index
from digsite.ingest.dedup import DEFAULT_MAX_DISTANCE
from digsite.ingest.pipeline import IngestStats, ingest
from digsite.store import (
    AuthorityStore,
    ChunkStore,
    CrawlStore,
    DocumentStore,
    connect,
)


def register(commands: Subparsers) -> None:
    corpus = corpus_options()

    crawl = commands.add_parser(
        "crawl", parents=[corpus], help="crawl a site and store its pages and links"
    )
    crawl.add_argument(
        "--seed",
        action="append",
        required=True,
        metavar="URL",
        help="start URL (repeatable)",
    )
    crawl.add_argument(
        "--prefix",
        action="append",
        default=[],
        metavar="URL",
        help="URL prefix the crawl may visit (repeatable; default: the hosts of the seeds)",
    )
    crawl.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="URL",
        help="URL prefix the crawl must stay out of (repeatable)",
    )
    crawl.add_argument("--max-pages", type=int, default=100, help="pages to store (default: 100)")
    crawl.add_argument("--max-depth", type=int, default=5, help="hops from a seed (default: 5)")
    crawl.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds between requests to the same host (default: 1.0)",
    )
    crawl.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="crawler User-Agent")
    crawl.set_defaults(handler=_crawl)

    ingest_parser = commands.add_parser(
        "ingest",
        parents=[corpus],
        help="extract the content of stored pages and mark duplicates",
    )
    ingest_parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        help="fingerprint bits two near-duplicates may differ in "
        f"(default: {DEFAULT_MAX_DISTANCE})",
    )
    ingest_parser.add_argument(
        "--force", action="store_true", help="extract every page again, not only the new ones"
    )
    ingest_parser.set_defaults(handler=_ingest)

    index = commands.add_parser(
        "index",
        parents=[corpus, analysis_options(), embedding_options()],
        help="split the documents into chunks, build the lexical and vector indexes "
        "and score the documents by their links",
    )
    index.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"largest chunk, in characters (default: {DEFAULT_MAX_CHARS})",
    )
    index.add_argument(
        "--no-embeddings",
        action="store_true",
        help="build only the lexical index; semantic search will not be available",
    )
    index.set_defaults(handler=_index)

    authority = commands.add_parser(
        "authority",
        parents=[corpus],
        help="show the documents that the links between documents make most important",
    )
    authority.add_argument("--limit", type=int, default=10, help="documents to show (default: 10)")
    authority.set_defaults(handler=_authority)

    stats = commands.add_parser("stats", parents=[corpus], help="show what the corpus contains")
    stats.set_defaults(handler=_stats)


def format_crawl_summary(stats: CrawlStats) -> str:
    summary = f"pages stored: {stats.saved}; links: {stats.links}; redirects: {stats.redirects}"
    if stats.skipped:
        reasons = ", ".join(f"{count} {reason}" for reason, count in stats.skipped.most_common())
        summary += f"; skipped: {reasons}"
    return summary


def format_ingest_summary(stats: IngestStats) -> str:
    documents = stats.documents
    return (
        f"pages extracted: {stats.extracted}; unique documents: {documents.unique}; "
        f"exact duplicates: {documents.exact_duplicates}; "
        f"near duplicates: {documents.near_duplicates}; empty: {documents.empty}"
    )


def _crawl(args: argparse.Namespace) -> int:
    config = CrawlConfig(
        seeds=tuple(args.seed),
        allowed_prefixes=tuple(args.prefix),
        excluded_prefixes=tuple(args.exclude),
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay,
        user_agent=args.user_agent,
    )
    with (
        closing(connect(corpus_path(args))) as connection,
        http_client(config) as client,
    ):
        try:
            crawler = Crawler(config, CrawlStore(connection), client)
        except ValueError as exc:
            return fail(str(exc), exit_code=2)
        try:
            stats = crawler.run()
        except KeyboardInterrupt:
            # Every page is committed as it is stored, so re-running resumes the crawl.
            print("\nInterrupted; what was downloaded so far is stored.", file=sys.stderr)
            return 130
    print(format_crawl_summary(stats))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    with closing(connect(path)) as connection:
        try:
            stats = ingest(
                CrawlStore(connection),
                DocumentStore(connection),
                max_distance=args.max_distance,
                force=args.force,
            )
        except KeyboardInterrupt:
            # Every document is committed as it is extracted, so re-running resumes.
            print("\nInterrupted; what was extracted so far is stored.", file=sys.stderr)
            return 130
    print(format_ingest_summary(stats))
    return 0


def _index(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    with closing(connect(path)) as connection:
        documents = DocumentStore(connection)
        if documents.stats().unique == 0:
            return fail("there are no documents to index; run 'digsite ingest' first")
        stats = build_index(
            documents,
            ChunkStore(connection),
            LexicalIndexStore(connection),
            analyzer_settings(args),
            embedder=None if args.no_embeddings else create_embedder(args.model),
            max_chars=args.max_chars,
        )
        authority = build_authority(CrawlStore(connection), documents, AuthorityStore(connection))
    summary = (
        f"documents: {stats.documents}; chunks: {stats.chunks}; "
        f"terms: {stats.terms}; postings: {stats.postings}; links: {authority.links}"
    )
    if not args.no_embeddings:
        summary += f"; vectors: {stats.embedded} computed, {stats.reused} reused"
    print(summary)
    return 0


def _authority(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    with closing(connect(path)) as connection:
        scores = AuthorityStore(connection).scores()
        documents = {
            document.page_id: document for document in DocumentStore(connection).documents()
        }
    if not scores:
        return fail("the documents have no link scores; run 'digsite index' first")
    # With no links at all every document would score this.
    uniform = 1 / len(scores)
    ranked = sorted(scores, key=lambda page_id: (-scores[page_id], page_id))
    for rank, page_id in enumerate(ranked[: args.limit], start=1):
        document = documents.get(page_id)
        title = document.title if document else "(document no longer stored)"
        print(f"{rank:>2}. {scores[page_id]:.4f}  {scores[page_id] / uniform:>5.1f}x  {title}")
        print(f"{' ' * 20}{document.url if document else ''}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    with closing(connect(path)) as connection:
        corpus = CrawlStore(connection).stats()
        documents = DocumentStore(connection).stats()
        loaded = LexicalIndexStore(connection).load()
        chunk_store = ChunkStore(connection)
        chunks = chunk_store.count()
        model = chunk_store.embedding_model()
        vectors = len(chunk_store.vectors(model)[0]) if model else 0
        scored = len(AuthorityStore(connection).scores())
    rows = [
        ("pages", corpus.pages),
        ("skipped urls", corpus.skipped),
        ("redirects", corpus.redirects),
        ("links", corpus.links),
        ("internal links", corpus.internal_links),
        ("compressed html bytes", corpus.compressed_html_bytes),
        ("unique documents", documents.unique),
        ("exact duplicates", documents.exact_duplicates),
        ("near duplicates", documents.near_duplicates),
        ("empty documents", documents.empty),
        ("chunks", chunks),
        ("indexed chunks", len(loaded[0]) if loaded else 0),
        ("index terms", loaded[0].vocabulary_size if loaded else 0),
        ("embedded chunks", vectors),
        ("link-scored documents", scored),
    ]
    for name, value in rows:
        print(f"{name:<24} {value:>10}")
    print(f"{'embedding model':<24} {model or '(none)'}")
    return 0
