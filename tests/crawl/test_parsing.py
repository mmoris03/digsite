from digsite.crawl.parsing import decode_body, parse_page
from digsite.models import Link

PAGE_URL = "https://example.com/docs/index.html"


def test_extracts_title_and_links_with_anchor_text() -> None:
    html = """
    <html><head><title>  Getting
      started </title></head>
    <body>
      <a href="/about">About <b>us</b></a>
      <a href="install.html">Installation</a>
    </body></html>
    """

    page = parse_page(html, PAGE_URL)

    assert page.title == "Getting started"
    assert page.links == [
        Link("https://example.com/about", "About us"),
        Link("https://example.com/docs/install.html", "Installation"),
    ]
    assert not page.noindex
    assert not page.nofollow


def test_drops_self_links_and_uncrawlable_schemes() -> None:
    html = """
    <a href="#section">in-page anchor</a>
    <a href="index.html#other">same page</a>
    <a href="mailto:info@example.com">mail</a>
    <a href="javascript:void(0)">js</a>
    <a href="">empty</a>
    <a name="no-href">no href</a>
    <a href="next.html">next</a>
    """

    page = parse_page(html, PAGE_URL)

    assert [link.url for link in page.links] == ["https://example.com/docs/next.html"]


def test_deduplicates_links_keeping_the_first_non_empty_anchor() -> None:
    html = """
    <a href="a.html"><img src="logo.png"></a>
    <a href="a.html#top">Page A</a>
    <a href="a.html">again</a>
    """

    page = parse_page(html, PAGE_URL)

    assert page.links == [Link("https://example.com/docs/a.html", "Page A")]


def test_base_href_changes_how_relative_links_resolve() -> None:
    html = '<head><base href="/other/base/"></head><a href="x.html">x</a>'

    page = parse_page(html, PAGE_URL)

    assert page.links == [Link("https://example.com/other/base/x.html", "x")]


def test_rel_nofollow_marks_only_that_link() -> None:
    html = '<a href="a.html" rel="external NOFOLLOW">a</a><a href="b.html">b</a>'

    page = parse_page(html, PAGE_URL)

    assert [(link.url[-6:], link.nofollow) for link in page.links] == [
        ("a.html", True),
        ("b.html", False),
    ]


def test_meta_robots_directives() -> None:
    html = '<meta name="ROBOTS" content="NoIndex, nofollow"><a href="a.html">a</a>'

    page = parse_page(html, PAGE_URL)

    assert page.noindex
    assert page.nofollow
    assert page.links[0].nofollow


def test_only_the_first_title_counts() -> None:
    html = "<title>Document</title><svg><title>Icon</title></svg>"

    assert parse_page(html, PAGE_URL).title == "Document"


def test_tolerates_broken_html() -> None:
    html = '<a href="a.html">never closed <div><a href="b.html">another'

    page = parse_page(html, PAGE_URL)

    assert [link.url[-6:] for link in page.links] == ["a.html", "b.html"]


def test_decode_prefers_header_charset() -> None:
    body = "año".encode("latin-1")

    assert decode_body(body, "text/html; charset=ISO-8859-1") == "año"


def test_decode_falls_back_to_meta_charset() -> None:
    body = '<meta charset="windows-1252"><p>año</p>'.encode("cp1252")

    assert "año" in decode_body(body, "text/html")


def test_decode_defaults_to_utf8_and_never_raises() -> None:
    assert decode_body("año".encode(), "text/html") == "año"
    assert decode_body(b"\xff\xfe", "text/html; charset=no-such-charset")
