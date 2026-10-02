"""HTTP download of a single page."""

from dataclasses import dataclass

import httpx

from digsite.models import SkipReason

_HTML_TYPES = ("text/html", "application/xhtml+xml")


@dataclass(frozen=True, slots=True)
class Fetched:
    """An HTML page was downloaded."""

    status: int
    content_type: str
    body: bytes


@dataclass(frozen=True, slots=True)
class Redirected:
    """The server answered with a redirect; `location` is the raw Location header."""

    status: int
    location: str


@dataclass(frozen=True, slots=True)
class Skipped:
    """The response was discarded, or there was no response (status 0)."""

    reason: SkipReason
    status: int = 0
    content_type: str = ""
    detail: str = ""


type FetchResult = Fetched | Redirected | Skipped


class Fetcher:
    """Requests URLs without following redirects and keeps only HTML bodies.

    Redirects are reported rather than followed so that the caller can apply the
    same checks (scope, robots.txt, pacing) to the target as to any other URL.
    """

    def __init__(self, client: httpx.Client, max_bytes: int) -> None:
        self._client = client
        self._max_bytes = max_bytes

    def fetch(self, url: str) -> FetchResult:
        try:
            with self._client.stream("GET", url, follow_redirects=False) as response:
                return self._read(response)
        except httpx.HTTPError as exc:
            return Skipped(SkipReason.NETWORK_ERROR, detail=f"{type(exc).__name__}: {exc}")

    def _read(self, response: httpx.Response) -> FetchResult:
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        if response.has_redirect_location:
            return Redirected(status, response.headers["location"])
        if status != 200:
            return Skipped(SkipReason.HTTP_ERROR, status, content_type, f"HTTP {status}")
        if not content_type.lower().startswith(_HTML_TYPES):
            return Skipped(SkipReason.NOT_HTML, status, content_type)

        # Read in chunks so an oversized response is abandoned, not downloaded whole.
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self._max_bytes:
                detail = f"larger than {self._max_bytes} bytes"
                return Skipped(SkipReason.TOO_LARGE, status, content_type, detail)
            chunks.append(chunk)
        return Fetched(status, content_type, b"".join(chunks))
