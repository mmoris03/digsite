"""A polite breadth-first crawler."""

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from digsite import __version__
from digsite.crawl.fetcher import Fetched, Fetcher, Redirected, Skipped
from digsite.crawl.frontier import Frontier
from digsite.crawl.parsing import decode_body, parse_page
from digsite.crawl.robots import RobotsCache
from digsite.crawl.throttle import HostThrottle
from digsite.crawl.urls import Scope, canonicalize, origin_of
from digsite.models import SkipReason
from digsite.store import CrawlStore

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = f"digsite/{__version__}"


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    """Settings for one crawl.

    Attributes:
        seeds: Start URLs.
        allowed_prefixes: URL prefixes the crawl may visit. When empty, the crawl
            is limited to the hosts of the seeds.
        excluded_prefixes: URL prefixes the crawl must stay out of, e.g. pages
            that are only indexes of other pages.
        max_pages: Maximum number of HTML pages to store in this run.
        max_depth: Maximum number of hops from a seed.
        delay_seconds: Minimum interval between two requests to the same host.
        timeout_seconds: Time limit per request.
        max_bytes: Maximum size of a page body.
        user_agent: How the crawler identifies itself to servers.
    """

    seeds: tuple[str, ...]
    allowed_prefixes: tuple[str, ...] = ()
    excluded_prefixes: tuple[str, ...] = ()
    max_pages: int = 100
    max_depth: int = 5
    delay_seconds: float = 1.0
    timeout_seconds: float = 15.0
    max_bytes: int = 5_000_000
    user_agent: str = DEFAULT_USER_AGENT


@dataclass(slots=True)
class CrawlStats:
    """What one crawler run did."""

    saved: int = 0
    links: int = 0
    redirects: int = 0
    skipped: Counter[SkipReason] = field(default_factory=Counter)


def http_client(config: CrawlConfig) -> httpx.Client:
    """Build an HTTP client that carries the crawl's User-Agent and timeout."""
    return httpx.Client(headers={"User-Agent": config.user_agent}, timeout=config.timeout_seconds)


class Crawler:
    """Visits, breadth-first, the pages reachable from the seeds.

    Stores each page's HTML and outgoing links. If the store already holds a
    previous crawl, the run picks up where that one stopped.

    Args:
        config: Crawl settings.
        store: Where pages and links are written.
        client: HTTP client; see `http_client`. The caller owns and closes it.
        sleep: Blocking sleep function; injected so tests do not actually wait.
        clock: Monotonic clock; injected for the same reason.

    Raises:
        ValueError: If a seed, every allowed prefix or an excluded prefix is not
            a crawlable URL.
    """

    def __init__(
        self,
        config: CrawlConfig,
        store: CrawlStore,
        client: httpx.Client,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._store = store
        self._seeds = _canonical_seeds(config.seeds)
        self._scope = Scope.from_seeds(
            self._seeds, config.allowed_prefixes, config.excluded_prefixes
        )
        self._robots = RobotsCache(client, config.user_agent)
        self._fetcher = Fetcher(client, config.max_bytes)
        self._throttle = HostThrottle(config.delay_seconds, sleep, clock)

    def run(self) -> CrawlStats:
        """Crawl until the frontier is empty or `max_pages` pages are stored."""
        stats = CrawlStats()
        frontier = Frontier(self._scope, self._config.max_depth, seen=self._store.known_urls())
        for seed in self._seeds:
            frontier.add(seed, 0)
        # Resuming: links a previous run discovered but never got to visit.
        for url, depth in self._store.pending_links():
            frontier.add(url, depth)

        while frontier and stats.saved < self._config.max_pages:
            url, depth = frontier.pop()
            self._visit(url, depth, frontier, stats)
        return stats

    def _visit(self, url: str, depth: int, frontier: Frontier, stats: CrawlStats) -> None:
        host = origin_of(url)
        allowed = self._robots.allowed(url)
        # Downloading robots.txt already counted as a request to this host.
        self._throttle.mark_first_contact(host)
        if not allowed:
            self._skip(url, depth, Skipped(SkipReason.ROBOTS), stats)
            return

        self._throttle.wait(host, min_delay=self._robots.crawl_delay(url) or 0.0)
        result = self._fetcher.fetch(url)
        self._throttle.mark(host)

        match result:
            case Fetched():
                self._save(url, depth, result, frontier, stats)
            case Redirected():
                self._redirect(url, depth, result, frontier, stats)
            case Skipped():
                self._skip(url, depth, result, stats)

    def _save(
        self, url: str, depth: int, fetched: Fetched, frontier: Frontier, stats: CrawlStats
    ) -> None:
        page = parse_page(decode_body(fetched.body, fetched.content_type), url)
        self._store.save_page(
            url,
            depth,
            fetched.body,
            status=fetched.status,
            content_type=fetched.content_type,
            title=page.title,
            noindex=page.noindex,
            links=page.links,
        )
        stats.saved += 1
        stats.links += len(page.links)
        logger.info(
            "[%d/%d] depth %d, %d links: %s",
            stats.saved,
            self._config.max_pages,
            depth,
            len(page.links),
            url,
        )

        for link in page.links:
            if not link.nofollow:
                frontier.add(link.url, depth + 1)

    def _redirect(
        self, url: str, depth: int, redirected: Redirected, frontier: Frontier, stats: CrawlStats
    ) -> None:
        target = canonicalize(redirected.location, base=url)
        if target is None or target == url:
            detail = f"Location: {redirected.location}"
            skipped = Skipped(SkipReason.BAD_REDIRECT, redirected.status, detail=detail)
            self._skip(url, depth, skipped, stats)
            return
        self._store.record_redirect(url, target)
        stats.redirects += 1
        # The target goes through the same checks as any other URL.
        frontier.add(target, depth, first=True)

    def _skip(self, url: str, depth: int, skipped: Skipped, stats: CrawlStats) -> None:
        self._store.record_skip(
            url,
            depth,
            skipped.reason,
            status=skipped.status,
            content_type=skipped.content_type,
            detail=skipped.detail,
        )
        stats.skipped[skipped.reason] += 1
        logger.info("skipped (%s): %s", skipped.reason, url)


def _canonical_seeds(seeds: tuple[str, ...]) -> tuple[str, ...]:
    canonical = [canonicalize(seed) for seed in seeds]
    invalid = [seed for seed, url in zip(seeds, canonical, strict=True) if url is None]
    if invalid or not seeds:
        raise ValueError(f"Seeds must be absolute HTTP(S) URLs; got {list(seeds)}")
    return tuple(url for url in canonical if url is not None)
