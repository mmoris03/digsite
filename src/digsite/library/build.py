"""Building a collection from a website: crawl, ingest and index, into a new file."""

import logging
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from digsite.crawl import CrawlConfig, Crawler, http_client
from digsite.crawl.urls import canonicalize
from digsite.embedding import Embedder
from digsite.index.analyzer import AnalyzerSettings, Language
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.pipeline import build_authority, build_index
from digsite.ingest.pipeline import ingest
from digsite.library.library import PARTIAL_SUFFIX
from digsite.models import CollectionInfo
from digsite.store import (
    AuthorityStore,
    ChunkStore,
    CollectionStore,
    CrawlStore,
    DocumentStore,
    connect,
)

logger = logging.getLogger(__name__)

MAX_PAGES = 500


class BuildStage(StrEnum):
    QUEUED = "queued"
    CRAWLING = "crawling"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


# Told the stage a build is in, and how far it has got: (stage, done, total).
type StageProgress = Callable[[BuildStage, int, int], None]


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """What to build a collection from.

    Attributes:
        url: Where the crawl starts. Only pages under the same directory are
            crawled: from https://example.com/docs/intro.html, those whose
            address starts with https://example.com/docs/.
        max_pages: Most pages to store.
        language: Language of the documents, for the lexical index's stop
            words and stemmer. None indexes plain words.
    """

    url: str
    max_pages: int = 50
    language: Language | None = None


class BuildError(Exception):
    """A collection could not be built. The message says why, for the person who asked."""


def crawl_scope(url: str) -> tuple[str, str]:
    """The canonical start URL of a crawl, and the prefix it stays under.

    Raises:
        BuildError: If the URL is not an absolute HTTP(S) address.
    """
    seed = canonicalize(url.strip())
    if seed is None:
        raise BuildError(f"{url!r} is not a web address (http:// or https://)")
    parts = urlsplit(seed)
    directory = parts.path[: parts.path.rfind("/") + 1] or "/"
    return seed, f"{parts.scheme}://{parts.netloc}{directory}"


def describe(crawl_store: CrawlStore, seed: str) -> CollectionInfo:
    """Describe a crawled collection by the title of the page it started at."""
    titles = {page.url: page.title for page in crawl_store.pages()}
    title = titles.get(seed, "").strip() or seed.split("://", 1)[-1].rstrip("/")
    return CollectionInfo(title=title, source=seed)


def build_collection(
    request: BuildRequest,
    path: Path,
    *,
    embedder: Embedder | None,
    client: httpx.Client | None = None,
    progress: StageProgress | None = None,
    delay_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> CollectionInfo:
    """Crawl a website and make it a collection, searchable as soon as the file exists.

    The collection is written to a file next to `path` and renamed to `path`
    only when complete, so that whoever lists collections never finds one
    half built. If the build fails, nothing is left behind.

    Args:
        request: What to build the collection from.
        path: The file of the new collection. It must not exist.
        embedder: Model to embed the passages with. None builds only the
            lexical index.
        client: HTTP client for the crawl. One is created and closed if none
            is given.
        progress: Told the stage of the build and how far it has got.
        delay_seconds: Interval between two requests to the website.
        sleep: Blocking sleep function; injected so that tests do not wait.

    Raises:
        BuildError: If the website gave nothing to index.
    """
    if path.exists():
        raise BuildError(f"there is already a collection at {path.name}")
    seed, prefix = crawl_scope(request.url)
    if not 1 <= request.max_pages <= MAX_PAGES:
        raise BuildError(f"the number of pages must be between 1 and {MAX_PAGES}")

    def report(stage: BuildStage) -> Callable[[int, int], None] | None:
        if progress is None:
            return None
        return lambda done, total: progress(stage, done, total)

    partial = path.with_name(path.name.removesuffix(".db") + PARTIAL_SUFFIX)
    partial.unlink(missing_ok=True)
    config = CrawlConfig(
        seeds=(seed,),
        allowed_prefixes=(prefix,),
        max_pages=request.max_pages,
        delay_seconds=delay_seconds,
    )
    try:
        with closing(connect(partial)) as connection:
            crawl_store = CrawlStore(connection)
            documents = DocumentStore(connection)
            if progress is not None:
                progress(BuildStage.CRAWLING, 0, request.max_pages)
            http = client or http_client(config)
            try:
                stats = Crawler(config, crawl_store, http, sleep=sleep).run(
                    progress=report(BuildStage.CRAWLING)
                )
            finally:
                if client is None:
                    http.close()
            if stats.saved == 0:
                reasons = ", ".join(
                    f"{count} {reason.replace('_', ' ')}" for reason, count in stats.skipped.items()
                )
                raise BuildError(
                    f"no page could be downloaded from {seed}"
                    + (f" ({reasons})" if reasons else "")
                )

            ingest(crawl_store, documents, progress=report(BuildStage.EXTRACTING))
            if documents.stats().unique == 0:
                raise BuildError(f"the pages under {prefix} have no text to index")

            if progress is not None:
                progress(BuildStage.INDEXING, 0, 0)
            build_index(
                documents,
                ChunkStore(connection),
                LexicalIndexStore(connection),
                AnalyzerSettings(language=request.language),
                embedder=embedder,
                progress=report(BuildStage.INDEXING),
            )
            build_authority(crawl_store, documents, AuthorityStore(connection))
            info = describe(crawl_store, seed)
            CollectionStore(connection).save(info)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    logger.info("built collection %s from %s", path.name, seed)
    return info
