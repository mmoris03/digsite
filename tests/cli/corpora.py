"""Sample corpora for command-line tests."""

from contextlib import closing
from pathlib import Path

from digsite.store import CrawlStore, connect
from fakes import article_html

SITE = "https://example.com"

CACHING = article_html(
    "Caching",
    "The cache stores the result of expensive computations so that repeated requests can be "
    "answered without doing the work again. Every entry is identified by a key and expires "
    "after a configurable amount of time.",
    "When the cache is full, the entry that was used least recently is removed first. This "
    "policy works well when recent requests are a good predictor of future ones.",
)
LOGGING = article_html(
    "Logging",
    "Every request is written to the access log with its duration, status code and the "
    "identity of the caller. Logs are rotated daily and kept for thirty days by default.",
    "Set the verbosity with the log level option. Debug output includes the full request "
    "body, so it should never be enabled on a production system that handles personal data.",
)


def make_corpus(data_dir: Path, pages: dict[str, str], *, collection: str = "corpus") -> None:
    """Create a corpus whose stored pages have the given HTML, by URL path."""
    with closing(connect(data_dir / f"{collection}.db")) as connection:
        store = CrawlStore(connection)
        for path, html in pages.items():
            store.save_page(f"{SITE}{path}", 0, html.encode(), content_type="text/html")
