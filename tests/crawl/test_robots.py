import httpx

from digsite.crawl.robots import RobotsCache
from fakes import FakeSite

AGENT = "digsite/0.1 (tests)"


def test_applies_rules_and_reads_crawl_delay() -> None:
    site = FakeSite({}, robots="User-agent: *\nDisallow: /private/\nCrawl-delay: 3\n")
    robots = RobotsCache(site.client, AGENT)

    assert robots.allowed("https://example.com/public/a.html")
    assert not robots.allowed("https://example.com/private/a.html")
    assert robots.crawl_delay("https://example.com/") == 3.0


def test_rules_addressed_to_this_agent_take_precedence() -> None:
    rules = "User-agent: digsite\nDisallow: /digsite-only/\n\nUser-agent: *\nDisallow: /\n"
    robots = RobotsCache(FakeSite({}, robots=rules).client, AGENT)

    assert robots.allowed("https://example.com/free")
    assert not robots.allowed("https://example.com/digsite-only/x")


def test_missing_robots_txt_allows_everything() -> None:
    robots = RobotsCache(FakeSite({}, robots=None).client, AGENT)

    assert robots.allowed("https://example.com/any/thing")
    assert robots.crawl_delay("https://example.com/") is None


def test_server_error_blocks_the_whole_site() -> None:
    site = FakeSite({"https://example.com/robots.txt": httpx.Response(503)})
    robots = RobotsCache(site.client, AGENT)

    assert not robots.allowed("https://example.com/")


def test_network_error_blocks_the_whole_site() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no connection", request=request)

    site = FakeSite({"https://example.com/robots.txt": fail})
    robots = RobotsCache(site.client, AGENT)

    assert not robots.allowed("https://example.com/")


def test_fetches_robots_txt_once_per_site() -> None:
    site = FakeSite({}, robots="User-agent: *\nDisallow:\n")
    robots = RobotsCache(site.client, AGENT)

    for path in ("a", "b", "c"):
        robots.allowed(f"https://example.com/{path}")
    robots.allowed("https://other.example.org/a")

    assert site.requested == [
        "https://example.com/robots.txt",
        "https://other.example.org/robots.txt",
    ]
