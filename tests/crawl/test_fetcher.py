import httpx

from digsite.crawl.fetcher import Fetched, Fetcher, Redirected, Skipped
from digsite.models import SkipReason
from fakes import HTML, FakeSite, html_page

URL = "https://example.com/page"


def fetch(route: httpx.Response, max_bytes: int = 1_000_000) -> object:
    site = FakeSite({URL: route})
    return Fetcher(site.client, max_bytes).fetch(URL)


def test_returns_the_body_of_an_html_page() -> None:
    result = fetch(httpx.Response(200, headers=HTML, content=b"<p>hello</p>"))

    assert result == Fetched(200, "text/html; charset=utf-8", b"<p>hello</p>")


def test_accepts_xhtml() -> None:
    headers = {"content-type": "application/xhtml+xml"}

    result = fetch(httpx.Response(200, headers=headers, content=b"<html/>"))

    assert isinstance(result, Fetched)


def test_reports_redirects_without_following_them() -> None:
    site = FakeSite(
        {
            URL: httpx.Response(301, headers={"location": "/moved"}),
            "https://example.com/moved": html_page(),
        }
    )

    result = Fetcher(site.client, 1_000_000).fetch(URL)

    assert result == Redirected(301, "/moved")
    assert site.requested == [URL]


def test_skips_responses_that_are_not_html() -> None:
    headers = {"content-type": "application/json"}

    result = fetch(httpx.Response(200, headers=headers, content=b"{}"))

    assert result == Skipped(SkipReason.NOT_HTML, 200, "application/json")


def test_skips_error_statuses() -> None:
    result = fetch(httpx.Response(500, headers=HTML, content=b"boom"))

    assert result == Skipped(SkipReason.HTTP_ERROR, 500, "text/html; charset=utf-8", "HTTP 500")


def test_skips_bodies_larger_than_the_limit() -> None:
    result = fetch(httpx.Response(200, headers=HTML, content=b"x" * 5000), max_bytes=1000)

    assert isinstance(result, Skipped)
    assert result.reason is SkipReason.TOO_LARGE


def test_network_errors_become_a_skip() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    site = FakeSite({URL: fail})

    result = Fetcher(site.client, 1_000_000).fetch(URL)

    assert result == Skipped(SkipReason.NETWORK_ERROR, detail="ReadTimeout: timed out")
