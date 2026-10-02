"""The queue of URLs waiting to be crawled."""

from collections import deque
from collections.abc import Iterable

from digsite.crawl.urls import Scope, has_skipped_extension


class Frontier:
    """A FIFO queue of URLs that accepts each URL at most once.

    First-in-first-out order makes the crawl breadth-first. A URL is accepted
    only if it is in scope, within the depth limit, looks like a page and has
    not been seen before.

    Args:
        scope: URL prefixes the crawl may visit.
        max_depth: Maximum number of hops from a seed.
        seen: URLs to treat as already visited, e.g. from a previous crawl.
    """

    def __init__(self, scope: Scope, max_depth: int, seen: Iterable[str] = ()) -> None:
        self._scope = scope
        self._max_depth = max_depth
        self._seen = set(seen)
        self._queue: deque[tuple[str, int]] = deque()

    def add(self, url: str, depth: int, *, first: bool = False) -> bool:
        """Queue a canonical URL if it is eligible.

        Args:
            url: Canonical URL.
            depth: Hops from the nearest seed.
            first: Put the URL at the front of the queue instead of the back.

        Returns:
            Whether the URL was queued.
        """
        if url in self._seen or depth > self._max_depth:
            return False
        if not self._scope.contains(url) or has_skipped_extension(url):
            return False
        self._seen.add(url)
        if first:
            self._queue.appendleft((url, depth))
        else:
            self._queue.append((url, depth))
        return True

    def pop(self) -> tuple[str, int]:
        """Remove and return the next (URL, depth) to visit."""
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)
