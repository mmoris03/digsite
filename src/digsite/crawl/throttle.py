"""Per-host request pacing."""

import time
from collections.abc import Callable


class HostThrottle:
    """Keeps a minimum interval between consecutive requests to the same host.

    Args:
        delay_seconds: Minimum interval between requests to one host.
        sleep: Blocking sleep function; injected so tests do not actually wait.
        clock: Monotonic clock; injected for the same reason.
    """

    def __init__(
        self,
        delay_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._delay = delay_seconds
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}

    def wait(self, host: str, min_delay: float = 0.0) -> None:
        """Block until the host may be contacted again.

        Args:
            host: Any stable key for the host; the crawler uses the URL origin.
            min_delay: A longer interval requested by the host itself.
        """
        last = self._last_request.get(host)
        if last is None:
            return
        remaining = max(self._delay, min_delay) - (self._clock() - last)
        if remaining > 0:
            self._sleep(remaining)

    def mark(self, host: str) -> None:
        """Record that a request to the host has just finished."""
        self._last_request[host] = self._clock()

    def mark_first_contact(self, host: str) -> None:
        """Record a request to the host unless one was already recorded."""
        self._last_request.setdefault(host, self._clock())
