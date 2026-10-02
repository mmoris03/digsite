"""Storage for crawl results: pages, their original HTML and the link graph.

The HTML is kept exactly as received (zlib-compressed) so that later pipeline
stages can be re-run without downloading anything again.
"""

import sqlite3
import zlib
from collections.abc import Iterable
from datetime import UTC, datetime

from digsite.models import CorpusStats, Edge, Link, Page, SkippedUrl, SkipReason

_UPSERT_PAGE = """
    INSERT INTO pages (url, depth, status, content_type, title, noindex, html,
                       skip_reason, skip_detail, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(url) DO UPDATE SET
        depth = excluded.depth, status = excluded.status,
        content_type = excluded.content_type, title = excluded.title,
        noindex = excluded.noindex, html = excluded.html,
        skip_reason = excluded.skip_reason, skip_detail = excluded.skip_detail,
        fetched_at = excluded.fetched_at
    RETURNING id
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CrawlStore:
    """Reads and writes crawl results on an open corpus database."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    def save_page(
        self,
        url: str,
        depth: int,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "",
        title: str = "",
        noindex: bool = False,
        links: Iterable[Link] = (),
    ) -> int:
        """Store a downloaded page together with its outgoing links.

        Saving the same URL again replaces the previous page and its links.

        Returns:
            The page id.
        """
        row = (
            url,
            depth,
            status,
            content_type,
            title,
            int(noindex),
            zlib.compress(body),
            None,
            "",
            _now(),
        )
        # One transaction: a page never exists without its links.
        with self._db:
            page_id: int = self._db.execute(_UPSERT_PAGE, row).fetchone()[0]
            self._db.execute("DELETE FROM links WHERE source_id = ?", (page_id,))
            self._db.executemany(
                "INSERT INTO links (source_id, target_url, anchor, nofollow) VALUES (?, ?, ?, ?)",
                ((page_id, link.url, link.anchor, int(link.nofollow)) for link in links),
            )
        return page_id

    def record_skip(
        self,
        url: str,
        depth: int,
        reason: SkipReason,
        *,
        status: int = 0,
        content_type: str = "",
        detail: str = "",
    ) -> None:
        """Remember that a URL was requested and why it was not stored."""
        row = (url, depth, status, content_type, "", 0, None, reason.value, detail, _now())
        with self._db:
            self._db.execute(_UPSERT_PAGE, row).fetchone()

    def record_redirect(self, source_url: str, target_url: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO redirects (source_url, target_url) VALUES (?, ?)",
                (source_url, target_url),
            )

    def known_urls(self) -> set[str]:
        """URLs already requested by previous crawls, whatever the outcome."""
        rows = self._db.execute("SELECT url FROM pages UNION SELECT source_url FROM redirects")
        return {row[0] for row in rows}

    def pending_links(self) -> list[tuple[str, int]]:
        """Followable link targets not requested yet, as (URL, depth) in crawl order."""
        rows = self._db.execute(
            """
            SELECT l.target_url, MIN(p.depth) + 1 AS depth
            FROM links l JOIN pages p ON p.id = l.source_id
            WHERE l.nofollow = 0
              AND l.target_url NOT IN (SELECT url FROM pages)
              AND l.target_url NOT IN (SELECT source_url FROM redirects)
            GROUP BY l.target_url
            ORDER BY depth, MIN(l.source_id)
            """
        )
        return [(row[0], row[1]) for row in rows]

    def html(self, url: str) -> bytes | None:
        """Return the original body of a stored page, or None if it is not stored."""
        row = self._db.execute("SELECT html FROM pages WHERE url = ?", (url,)).fetchone()
        return zlib.decompress(row[0]) if row and row[0] is not None else None

    def pages(self) -> list[Page]:
        """Stored pages, in download order."""
        rows = self._db.execute(
            "SELECT id, url, depth, title, noindex, content_type FROM pages "
            "WHERE html IS NOT NULL ORDER BY id"
        )
        return [Page(row[0], row[1], row[2], row[3], bool(row[4]), row[5]) for row in rows]

    def skipped(self) -> list[SkippedUrl]:
        """URLs that were requested but not stored, in request order."""
        rows = self._db.execute(
            "SELECT url, skip_reason, status, skip_detail FROM pages WHERE html IS NULL ORDER BY id"
        )
        return [SkippedUrl(row[0], SkipReason(row[1]), row[2], row[3]) for row in rows]

    def edges(self) -> list[Edge]:
        """Every link in the corpus, including those pointing outside of it."""
        rows = self._db.execute(
            """
            SELECT p.url, l.target_url, l.nofollow
            FROM links l JOIN pages p ON p.id = l.source_id
            ORDER BY l.source_id, l.target_url
            """
        )
        return [Edge(row[0], row[1], bool(row[2])) for row in rows]

    def redirects(self) -> dict[str, str]:
        return dict(self._db.execute("SELECT source_url, target_url FROM redirects"))

    def stats(self) -> CorpusStats:
        def count(sql: str) -> int:
            value: int = self._db.execute(sql).fetchone()[0]
            return value

        return CorpusStats(
            pages=count("SELECT COUNT(*) FROM pages WHERE html IS NOT NULL"),
            skipped=count("SELECT COUNT(*) FROM pages WHERE html IS NULL"),
            redirects=count("SELECT COUNT(*) FROM redirects"),
            links=count("SELECT COUNT(*) FROM links"),
            internal_links=count(
                "SELECT COUNT(*) FROM links "
                "WHERE target_url IN (SELECT url FROM pages WHERE html IS NOT NULL)"
            ),
            compressed_html_bytes=count("SELECT COALESCE(SUM(LENGTH(html)), 0) FROM pages"),
        )
