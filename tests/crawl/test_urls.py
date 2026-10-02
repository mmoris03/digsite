import pytest

from digsite.crawl.urls import Scope, canonicalize, has_skipped_extension, origin_of

BASE = "https://example.com/docs/intro.html"


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        # A fragment does not change the resource: it is the same page.
        ("#history", "https://example.com/docs/intro.html"),
        ("https://example.com/docs/intro.html#main-content", "https://example.com/docs/intro.html"),
        # Relative references, with and without a leading slash.
        ("/about", "https://example.com/about"),
        ("page2.html", "https://example.com/docs/page2.html"),
        ("../up.html", "https://example.com/up.html"),
        ("//cdn.example.com/x", "https://cdn.example.com/x"),
        # Lower-case scheme and host; default port and credentials dropped.
        ("HTTPS://Example.COM:443/A", "https://example.com/A"),
        ("http://example.com:8080/", "http://example.com:8080/"),
        ("https://example.com", "https://example.com/"),
        ("https://user:secret@example.com/x", "https://example.com/x"),
        # Dot segments.
        ("https://example.com/a/./b/../c", "https://example.com/a/c"),
        ("https://example.com/a/b/..", "https://example.com/a/"),
        # Sorted query, tracking parameters removed.
        ("https://example.com/s?b=2&a=1", "https://example.com/s?a=1&b=2"),
        ("https://example.com/s?utm_source=x&q=1&fbclid=abc", "https://example.com/s?q=1"),
        ("https://example.com/s?utm_campaign=x", "https://example.com/s"),
        # Normalised percent-encoding.
        ("https://example.com/%7Euser/a%2fb", "https://example.com/~user/a%2Fb"),
        ("https://example.com/with space/ñ", "https://example.com/with%20space/%C3%B1"),
    ],
)
def test_canonicalize(href: str, expected: str) -> None:
    assert canonicalize(href, base=BASE) == expected


@pytest.mark.parametrize(
    "href",
    [
        "mailto:info@example.com",
        "javascript:void(0)",
        "tel:+34600000000",
        "data:text/plain;base64,QQ==",
        "ftp://example.com/file",
        "",
        "   ",
        "http://[::1",
        "https://example.com:99999/",
    ],
)
def test_canonicalize_rejects_what_cannot_be_crawled(href: str) -> None:
    assert canonicalize(href, base=BASE) is None


def test_canonicalize_is_idempotent() -> None:
    once = canonicalize("HTTP://Example.com/a/../b%7e?z=1&a=%20#frag")
    assert once is not None
    assert canonicalize(once) == once


def test_relative_url_without_base_is_rejected() -> None:
    assert canonicalize("page2.html") is None


@pytest.mark.parametrize(
    ("url", "skipped"),
    [
        ("https://example.com/manual.pdf", True),
        ("https://example.com/logo.PNG", True),
        ("https://example.com/page.html", False),
        ("https://example.com/v1.2/guide", False),
        ("https://example.com/archive.zip/readme", False),
        ("https://example.com/", False),
    ],
)
def test_has_skipped_extension(url: str, skipped: bool) -> None:
    assert has_skipped_extension(url) is skipped


def test_origin_of() -> None:
    assert origin_of("https://example.com:8443/a/b?c=1") == "https://example.com:8443"


def test_scope_defaults_to_seed_hosts() -> None:
    scope = Scope.from_seeds(("https://example.com/docs/", "https://other.org/x"))

    assert scope.contains("https://example.com/anything")
    assert scope.contains("https://other.org/")
    assert not scope.contains("https://example.com.evil.org/")
    assert not scope.contains("http://example.com/")


def test_scope_with_explicit_prefixes() -> None:
    scope = Scope.from_seeds(("https://example.com/",), ("https://example.com/es/3/",))

    assert scope.contains("https://example.com/es/3/tutorial/")
    assert not scope.contains("https://example.com/en/3/tutorial/")


def test_scope_needs_at_least_one_valid_prefix() -> None:
    with pytest.raises(ValueError, match="No valid URL prefix"):
        Scope.from_seeds(("https://example.com/",), ("mailto:nobody",))


def test_scope_leaves_out_excluded_prefixes() -> None:
    scope = Scope.from_seeds(
        ("https://example.com/docs/",),
        excluded=("https://example.com/docs/genindex", "HTTPS://Example.com/docs/search.html"),
    )

    assert scope.contains("https://example.com/docs/tutorial.html")
    assert not scope.contains("https://example.com/docs/genindex.html")
    assert not scope.contains("https://example.com/docs/genindex-A.html")
    assert not scope.contains("https://example.com/docs/search.html")


def test_scope_rejects_an_excluded_prefix_that_is_not_a_url() -> None:
    with pytest.raises(ValueError, match="Excluded prefixes"):
        Scope.from_seeds(("https://example.com/",), excluded=("genindex",))
