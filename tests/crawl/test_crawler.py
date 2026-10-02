from collections import Counter
from typing import Any

import httpx
import pytest

from digsite.crawl import CrawlConfig, Crawler, CrawlStats
from digsite.models import Edge, SkippedUrl, SkipReason
from digsite.store import CrawlStore
from fakes import HTML, FakeClock, FakeSite, html_page

SITE = "https://example.com"


def crawl(
    site: FakeSite,
    store: CrawlStore,
    clock: FakeClock,
    seeds: tuple[str, ...] = (f"{SITE}/",),
    **options: Any,
) -> CrawlStats:
    config = CrawlConfig(seeds=seeds, **options)
    return Crawler(config, store, site.client, sleep=clock.sleep, clock=clock).run()


def stored_urls(store: CrawlStore) -> list[str]:
    return [page.url for page in store.pages()]


def test_visits_pages_breadth_first_and_stores_the_link_graph(
    store: CrawlStore, clock: FakeClock
) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/a", "/b", title="Home"),
            f"{SITE}/a": html_page("/c"),
            f"{SITE}/b": html_page("/"),
            f"{SITE}/c": html_page(),
        }
    )

    stats = crawl(site, store, clock)

    assert site.requested == [
        f"{SITE}/robots.txt",
        f"{SITE}/",
        f"{SITE}/a",
        f"{SITE}/b",
        f"{SITE}/c",
    ]
    assert stats == CrawlStats(saved=4, links=4)
    assert [(page.url, page.depth) for page in store.pages()] == [
        (f"{SITE}/", 0),
        (f"{SITE}/a", 1),
        (f"{SITE}/b", 1),
        (f"{SITE}/c", 2),
    ]
    assert store.pages()[0].title == "Home"
    assert store.edges() == [
        Edge(f"{SITE}/", f"{SITE}/a", False),
        Edge(f"{SITE}/", f"{SITE}/b", False),
        Edge(f"{SITE}/a", f"{SITE}/c", False),
        Edge(f"{SITE}/b", f"{SITE}/", False),
    ]
    html = store.html(f"{SITE}/")
    assert html is not None
    assert b"<title>Home</title>" in html


def test_the_same_page_under_different_fragments_is_downloaded_once(
    store: CrawlStore, clock: FakeClock
) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("#history", "#figures", "/#clients", "/about/#history"),
            f"{SITE}/about/": html_page("#team", "/"),
        }
    )

    crawl(site, store, clock)

    assert site.pages_requested == [f"{SITE}/", f"{SITE}/about/"]


def test_hrefs_that_are_not_pages_do_not_break_the_crawl(
    store: CrawlStore, clock: FakeClock
) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page(
                "mailto:info@example.com",
                "javascript:void(0)",
                "tel:+34600",
                "page2.html",
                "/manual.pdf",
            ),
            f"{SITE}/page2.html": html_page(),
        }
    )

    stats = crawl(site, store, clock)

    assert site.pages_requested == [f"{SITE}/", f"{SITE}/page2.html"]
    assert stats.saved == 2
    assert not stats.skipped


def test_stops_after_storing_max_pages(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {f"{SITE}/": html_page("/1", "/2", "/3")} | {f"{SITE}/{n}": html_page() for n in "123"}
    )

    stats = crawl(site, store, clock, max_pages=2)

    assert stats.saved == 2
    assert stored_urls(store) == [f"{SITE}/", f"{SITE}/1"]


def test_does_not_go_deeper_than_max_depth(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/1"),
            f"{SITE}/1": html_page("/2"),
            f"{SITE}/2": html_page("/3"),
        }
    )

    crawl(site, store, clock, max_depth=1)

    assert stored_urls(store) == [f"{SITE}/", f"{SITE}/1"]


def test_stays_in_scope_but_records_external_links(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite({f"{SITE}/": html_page("https://external.org/page", "/inside")})

    crawl(site, store, clock)

    assert "https://external.org/page" not in site.requested
    assert Edge(f"{SITE}/", "https://external.org/page", False) in store.edges()


def test_prefix_restricts_the_crawl_to_a_section(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {
            f"{SITE}/es/3/": html_page("/es/3/tutorial", "/en/3/tutorial"),
            f"{SITE}/es/3/tutorial": html_page(),
            f"{SITE}/en/3/tutorial": html_page(),
        }
    )

    crawl(site, store, clock, seeds=(f"{SITE}/es/3/",), allowed_prefixes=(f"{SITE}/es/3/",))

    assert stored_urls(store) == [f"{SITE}/es/3/", f"{SITE}/es/3/tutorial"]


def test_excluded_prefixes_are_not_visited_but_stay_in_the_graph(
    store: CrawlStore, clock: FakeClock
) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/genindex.html", "/genindex-A.html", "/tutorial"),
            f"{SITE}/genindex.html": html_page("/hidden"),
            f"{SITE}/genindex-A.html": html_page(),
            f"{SITE}/tutorial": html_page(),
        }
    )

    crawl(site, store, clock, excluded_prefixes=(f"{SITE}/genindex",))

    assert site.pages_requested == [f"{SITE}/", f"{SITE}/tutorial"]
    assert Edge(f"{SITE}/", f"{SITE}/genindex.html", False) in store.edges()


def test_respects_robots_txt(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/private/a", "/public"),
            f"{SITE}/public": html_page(),
            f"{SITE}/private/a": html_page(),
        },
        robots="User-agent: *\nDisallow: /private/\n",
    )

    stats = crawl(site, store, clock)

    assert f"{SITE}/private/a" not in site.requested
    assert stats.skipped == Counter({SkipReason.ROBOTS: 1})
    assert store.skipped() == [SkippedUrl(f"{SITE}/private/a", SkipReason.ROBOTS, 0, "")]


def test_does_not_follow_nofollow_links(store: CrawlStore, clock: FakeClock) -> None:
    page_with_rel = httpx.Response(
        200, headers=HTML, content=b'<a href="/a" rel="nofollow">a</a><a href="/b">b</a>'
    )
    site = FakeSite(
        {
            f"{SITE}/": page_with_rel,
            f"{SITE}/a": html_page(),
            f"{SITE}/b": html_page("/c", head='<meta name="robots" content="nofollow">'),
            f"{SITE}/c": html_page(),
        }
    )

    crawl(site, store, clock)

    assert site.pages_requested == [f"{SITE}/", f"{SITE}/b"]
    assert Edge(f"{SITE}/", f"{SITE}/a", True) in store.edges()
    assert Edge(f"{SITE}/b", f"{SITE}/c", True) in store.edges()


def test_responses_that_are_not_html_do_not_use_up_the_page_budget(
    store: CrawlStore, clock: FakeClock
) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/data", "/page"),
            f"{SITE}/data": httpx.Response(
                200, headers={"content-type": "application/json"}, content=b"{}"
            ),
            f"{SITE}/page": html_page(),
        }
    )

    stats = crawl(site, store, clock, max_pages=2)

    assert stats.saved == 2
    assert stats.skipped == Counter({SkipReason.NOT_HTML: 1})
    assert store.skipped() == [
        SkippedUrl(f"{SITE}/data", SkipReason.NOT_HTML, 200, ""),
    ]


def test_http_and_network_errors_are_recorded_and_not_fatal(
    store: CrawlStore, clock: FakeClock
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    site = FakeSite(
        {
            f"{SITE}/": html_page("/down", "/missing", "/fine"),
            f"{SITE}/down": fail,
            f"{SITE}/fine": html_page(),
        }
    )

    stats = crawl(site, store, clock)

    assert stats.saved == 2
    assert stats.skipped == Counter({SkipReason.NETWORK_ERROR: 1, SkipReason.HTTP_ERROR: 1})
    assert store.skipped() == [
        SkippedUrl(f"{SITE}/down", SkipReason.NETWORK_ERROR, 0, "ConnectTimeout: timed out"),
        SkippedUrl(f"{SITE}/missing", SkipReason.HTTP_ERROR, 404, "HTTP 404"),
    ]


def test_skips_pages_larger_than_max_bytes(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/huge"),
            f"{SITE}/huge": httpx.Response(200, headers=HTML, content=b"x" * 5000),
        }
    )

    stats = crawl(site, store, clock, max_bytes=1000)

    assert stats.saved == 1
    assert [skipped.reason for skipped in store.skipped()] == [SkipReason.TOO_LARGE]


def test_follows_redirects_inside_the_scope_and_records_them(
    store: CrawlStore, clock: FakeClock
) -> None:
    site = FakeSite(
        {
            f"{SITE}/": html_page("/old", "/out", "/other"),
            f"{SITE}/old": httpx.Response(301, headers={"location": "/new#section"}),
            f"{SITE}/new": html_page(title="New"),
            f"{SITE}/out": httpx.Response(302, headers={"location": "https://external.org/"}),
            f"{SITE}/other": html_page(),
        }
    )

    stats = crawl(site, store, clock)

    # A redirect target is visited next, ahead of the rest of the queue.
    assert site.pages_requested == [
        f"{SITE}/",
        f"{SITE}/old",
        f"{SITE}/new",
        f"{SITE}/out",
        f"{SITE}/other",
    ]
    assert stats.redirects == 2
    assert store.redirects() == {
        f"{SITE}/old": f"{SITE}/new",
        f"{SITE}/out": "https://external.org/",
    }
    assert stored_urls(store) == [f"{SITE}/", f"{SITE}/new", f"{SITE}/other"]


def test_redirect_loops_terminate(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {
            f"{SITE}/": httpx.Response(302, headers={"location": "/b"}),
            f"{SITE}/b": httpx.Response(302, headers={"location": "/"}),
        }
    )

    stats = crawl(site, store, clock)

    assert stats.saved == 0
    assert stats.redirects == 2


def test_a_redirect_to_itself_is_recorded_as_a_skip(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite({f"{SITE}/": httpx.Response(302, headers={"location": "/#top"})})

    stats = crawl(site, store, clock)

    assert stats.skipped == Counter({SkipReason.BAD_REDIRECT: 1})
    assert store.skipped() == [
        SkippedUrl(f"{SITE}/", SkipReason.BAD_REDIRECT, 302, "Location: /#top")
    ]


def test_waits_between_requests_to_the_same_host(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {f"{SITE}/": html_page("/a", "/b"), f"{SITE}/a": html_page(), f"{SITE}/b": html_page()}
    )

    crawl(site, store, clock, delay_seconds=2.0)

    # One wait before each page; the first one follows the robots.txt download.
    assert clock.sleeps == [2.0, 2.0, 2.0]


def test_honours_a_longer_crawl_delay_from_robots_txt(store: CrawlStore, clock: FakeClock) -> None:
    site = FakeSite(
        {f"{SITE}/": html_page("/a"), f"{SITE}/a": html_page()},
        robots="User-agent: *\nCrawl-delay: 5\n",
    )

    crawl(site, store, clock, delay_seconds=1.0)

    assert clock.sleeps == [5.0, 5.0]


def test_a_second_run_resumes_without_downloading_again(
    store: CrawlStore, clock: FakeClock
) -> None:
    routes = {f"{SITE}/": html_page("/1", "/2", "/3")} | {f"{SITE}/{n}": html_page() for n in "123"}
    first = FakeSite(routes)
    crawl(first, store, clock, max_pages=2)

    second = FakeSite(routes)
    stats = crawl(second, store, clock, max_pages=10)

    assert second.pages_requested == [f"{SITE}/2", f"{SITE}/3"]
    assert stats.saved == 2
    assert stored_urls(store) == [f"{SITE}/", f"{SITE}/1", f"{SITE}/2", f"{SITE}/3"]


@pytest.mark.parametrize("seeds", [("mailto:nobody@example.com",), ("/relative",), ()])
def test_rejects_invalid_seeds(store: CrawlStore, clock: FakeClock, seeds: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="Seeds must be absolute"):
        crawl(FakeSite({}), store, clock, seeds=seeds)
