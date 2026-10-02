"""The ingest stage: extract the content of stored pages and mark duplicates."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from digsite.crawl.parsing import decode_body
from digsite.ingest.content import Content, extract_content
from digsite.ingest.dedup import DEFAULT_MAX_DISTANCE, content_hash, find_duplicates, fingerprint
from digsite.models import DocumentStats, Page
from digsite.store import CrawlStore, DocumentStore
from digsite.text import tokenize

logger = logging.getLogger(__name__)

type Extractor = Callable[[str, str], Content | None]

_PROGRESS_EVERY = 25


@dataclass(frozen=True, slots=True)
class IngestStats:
    """What one ingest run did.

    Attributes:
        extracted: Pages whose content was extracted by this run.
        documents: State of the whole corpus after the run.
    """

    extracted: int
    documents: DocumentStats


def ingest(
    crawl_store: CrawlStore,
    document_store: DocumentStore,
    *,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    force: bool = False,
    extract: Extractor = extract_content,
) -> IngestStats:
    """Turn stored pages into documents and mark the duplicates among them.

    Extraction is the slow part, so it only runs for pages that have no
    document yet. Duplicate marks are cheap and depend on the whole corpus, so
    they are recomputed on every run.

    Args:
        crawl_store: Source of pages and their HTML.
        document_store: Destination of documents and duplicate marks.
        max_distance: Largest Hamming distance between near-duplicates.
        force: Extract every page again, discarding existing documents.
        extract: Function that extracts the main content of an HTML page.
    """
    if force:
        document_store.clear()

    done = document_store.page_ids()
    pending = [page for page in crawl_store.pages() if page.id not in done and not page.noindex]
    for number, page in enumerate(pending, start=1):
        _extract_page(page, crawl_store, document_store, extract)
        if number % _PROGRESS_EVERY == 0 or number == len(pending):
            logger.info("extracted %d/%d pages", number, len(pending))

    duplicates = find_duplicates(document_store.fingerprints(), max_distance=max_distance)
    document_store.set_duplicates(duplicates)
    for duplicate in document_store.duplicates():
        logger.debug(
            "%s duplicate: %s -> %s", duplicate.kind, duplicate.url, duplicate.canonical_url
        )
    return IngestStats(extracted=len(pending), documents=document_store.stats())


def _extract_page(
    page: Page, crawl_store: CrawlStore, document_store: DocumentStore, extract: Extractor
) -> None:
    html = crawl_store.html(page.url)
    content = extract(decode_body(html, page.content_type), page.url) if html else None
    text = content.text if content else ""
    if not tokenize(text):  # nothing but punctuation is as good as nothing
        text = ""
    title = (content.title if content else "") or page.title
    document_store.save(
        page.id, title, text, content_hash(text), fingerprint(text) if text else None
    )
