import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from digsite.models import CorpusStats, Edge, Link, Page, SkippedUrl, SkipReason
from digsite.store import CrawlStore, connect

SITE = "https://example.com"


def test_html_round_trip(store: CrawlStore) -> None:
    body = "<html><body>año</body></html>".encode()

    store.save_page(f"{SITE}/", 0, body, title="Home")

    assert store.html(f"{SITE}/") == body
    assert store.html(f"{SITE}/missing") is None
    assert store.pages() == [Page(1, f"{SITE}/", 0, "Home", False)]


def test_a_page_is_stored_together_with_its_links(store: CrawlStore) -> None:
    links = [Link(f"{SITE}/a", "A"), Link("https://external.org/", "out", nofollow=True)]

    store.save_page(f"{SITE}/", 0, b"<p>home</p>", links=links)

    assert store.edges() == [
        Edge(f"{SITE}/", f"{SITE}/a", False),
        Edge(f"{SITE}/", "https://external.org/", True),
    ]


def test_saving_a_page_again_replaces_it_and_its_links(store: CrawlStore) -> None:
    first = store.save_page(f"{SITE}/", 0, b"<p>v1</p>", links=[Link(f"{SITE}/old")])

    second = store.save_page(f"{SITE}/", 0, b"<p>v2</p>", links=[Link(f"{SITE}/new")])

    assert first == second
    assert store.html(f"{SITE}/") == b"<p>v2</p>"
    assert store.edges() == [Edge(f"{SITE}/", f"{SITE}/new", False)]


def test_a_failed_save_stores_neither_the_page_nor_its_links(store: CrawlStore) -> None:
    def links_that_fail() -> Iterator[Link]:
        yield Link(f"{SITE}/a")
        raise RuntimeError("parser crashed")

    with pytest.raises(RuntimeError):
        store.save_page(f"{SITE}/", 0, b"<p>home</p>", links=links_that_fail())

    assert store.pages() == []
    assert store.edges() == []


def test_skipped_urls_keep_the_reason(store: CrawlStore) -> None:
    store.record_skip(
        f"{SITE}/data.json", 1, SkipReason.NOT_HTML, status=200, content_type="application/json"
    )
    store.record_skip(f"{SITE}/down", 1, SkipReason.NETWORK_ERROR, detail="ConnectError")

    assert store.pages() == []
    assert store.html(f"{SITE}/data.json") is None
    assert store.skipped() == [
        SkippedUrl(f"{SITE}/data.json", SkipReason.NOT_HTML, 200, ""),
        SkippedUrl(f"{SITE}/down", SkipReason.NETWORK_ERROR, 0, "ConnectError"),
    ]


def test_a_skipped_url_can_later_be_stored(store: CrawlStore) -> None:
    store.record_skip(f"{SITE}/", 0, SkipReason.NETWORK_ERROR, detail="ConnectError")

    store.save_page(f"{SITE}/", 0, b"<p>ok</p>")

    assert store.skipped() == []
    assert store.html(f"{SITE}/") == b"<p>ok</p>"


def test_pending_links_are_unvisited_followable_targets_in_crawl_order(
    store: CrawlStore,
) -> None:
    store.save_page(
        f"{SITE}/",
        0,
        b"",
        links=[
            Link(f"{SITE}/a", "already visited"),
            Link(f"{SITE}/b", "b"),
            Link(f"{SITE}/old", "redirects"),
            Link(f"{SITE}/secret", "do not follow", nofollow=True),
        ],
    )
    store.save_page(
        f"{SITE}/a", 1, b"", links=[Link(f"{SITE}/c", "c"), Link(f"{SITE}/b", "b again")]
    )
    store.record_redirect(f"{SITE}/old", f"{SITE}/a")

    # /b is reachable from the home page, so its depth is 1 and it comes before /c.
    assert store.pending_links() == [(f"{SITE}/b", 1), (f"{SITE}/c", 2)]
    assert store.known_urls() == {f"{SITE}/", f"{SITE}/a", f"{SITE}/old"}
    assert store.redirects() == {f"{SITE}/old": f"{SITE}/a"}


def test_stats(store: CrawlStore) -> None:
    store.save_page(
        f"{SITE}/", 0, b"<p>home</p>", links=[Link(f"{SITE}/a"), Link("https://external.org/")]
    )
    store.save_page(f"{SITE}/a", 1, b"<p>a</p>")
    store.record_skip(f"{SITE}/x.json", 1, SkipReason.NOT_HTML, status=200)
    store.record_redirect(f"{SITE}/old", f"{SITE}/a")

    stats = store.stats()

    assert stats == CorpusStats(
        pages=2,
        skipped=1,
        redirects=1,
        links=2,
        internal_links=1,
        compressed_html_bytes=stats.compressed_html_bytes,
    )
    assert stats.compressed_html_bytes > 0


def test_data_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.db"
    with closing(connect(path)) as connection:
        CrawlStore(connection).save_page(f"{SITE}/", 0, b"<p>home</p>", title="Home")

    with closing(connect(path)) as reopened:
        assert CrawlStore(reopened).pages() == [Page(1, f"{SITE}/", 0, "Home", False)]


def test_two_stores_can_share_one_connection(connection: sqlite3.Connection) -> None:
    CrawlStore(connection).save_page(f"{SITE}/", 0, b"<p>home</p>")

    assert CrawlStore(connection).html(f"{SITE}/") == b"<p>home</p>"
